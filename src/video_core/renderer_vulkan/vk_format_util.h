// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <span>
#include <bit>
#include <string_view>
#include <array>
#include <cstring>
#include <bitset>
#include <type_traits>
#include <vulkan/vulkan_format_traits.hpp>
#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore::Vulkan {

enum class SIMD : u8 {
    None = 0,
    SSE4 = 1,
    AVX2 = 2,
    NEON = 3
};

/**
 * A Pixel holds a pixel value or a SIMD lane holding multiple "real" pixels
 */
#pragma pack(1)
template <u8 bytes, SIMD simd = SIMD::None>
struct Pixel {
    using StorageType = std::conditional_t<bytes <= 1, u8,
                        std::conditional_t<bytes <= 2, u16,
                        std::conditional_t<bytes <= 4, u32, u64>>>;
    Pixel() = default;

    // Memory load/store
    constexpr void Load(u8* memory) {
        std::memcpy(&storage, memory, bytes);
    }

    constexpr void Store(u8* memory) const {
        std::memcpy(memory, &storage, bytes);
    }

    // Returns the number of bytes until the next pixel
    constexpr u8 GetStride() const {
        return bytes;
    }

    // Bitwise operators
    constexpr Pixel RotateRight(int n) const {
        return std::rotr(storage, n);
    }

    constexpr StorageType operator & (const StorageType mask) const {
        return storage & mask;
    }

    constexpr StorageType operator | (const StorageType mask) const {
        return storage | mask;
    }

    constexpr StorageType operator >>(const int n) const {
        return storage >> n;
    }

    constexpr StorageType operator <<(const int n) const {
        return storage << n;
    }

private:
    StorageType storage;
};
#pragma pack()

/**
 * Information about a pixel format
 */
template <u8 Components>
struct FormatInfo {
    constexpr FormatInfo(vk::Format format) {
        for (int i = 0; i < components; i++) {
            name[i] = vk::componentName(format, i)[0];
            is_float[i] = std::string_view{vk::componentNumericFormat(format, i)}
                        == "SFLOAT";
            bits[i] = vk::componentBits(format, i);
            bit_offset[i] = (i > 0 ? bit_offset[i - 1] + bits[i - 1] : 0);
        }

        bytes = (format == vk::Format::eD32SfloatS8Uint ? 8 :
                 vk::blockSize(format));
    }

    static constexpr u32 components = Components;
    std::array<char, components> name;
    std::array<bool, components> is_float;
    std::array<u8, components> bit_offset;
    std::array<u8, components> bits;
    u8 bytes; // This includes the padding in D32S8
};

/**
 * Represents a mapping of components from one format to another
 */
template <FormatInfo source, FormatInfo dest>
struct Mapping {
    static constexpr u32 component_map_bits = 4;
    static constexpr u32 component_map_mask = (1 << component_map_bits) - 1;

    constexpr Mapping() {
        for (int i = 0; i < source.names.size(); i++) {
            constexpr char source_name = source.names[i];
            for (u8 j = 0; j < dest.names.size(); j++) {
                constexpr char dest_name = dest.names[j];
                if constexpr (source_name == dest_name) {
                    storage |= ((j & component_map_mask) << component_map_bits * i);
                    break;
                }
            }
        }
    }

    constexpr u8 GetMapping(const int component) {
        return (storage >> (component * component_map_bits)) & component_map_mask;
    }

    // Returns the number of bits to rotate a pixel to the right
    // to match the mapping of the destiation format. If it's not
    // possible returns -1
    constexpr s32 TestMappingRotation() {
        constexpr u16 identity = 0x3210;

        u32 total_bits_rotated = 0;
        auto test_rotation = [&](s32 i) -> bool {
            return (storage == std::rotr(identity, i * component_map_bits));
        };

        for (s32 rot = 0; rot < 4; rot++) {
            if (test_rotation(rot)) {
                return total_bits_rotated;
            }

            total_bits_rotated += source.bits[rot];
        }

        return -1;
    }

    // Returns true if the each component of the source format has the
    // same bit-width as the mapped destination format component
    constexpr bool AreBitwiseEqual() {
        bool result = source.bytes == dest.bytes;
        for (int i = 0; i < source.components; i++) {
            result &= (source.bits[i] == dest.bits[GetMapping(i)]);
        }

        return result;
    }

private:
    // Since there are at most 4 components we can use 4 bits for each component
    u16 storage = 0xFFFF;
};

// Allows for loop like iteration at compile time
template <auto Start, auto End, class F>
constexpr void ForEach(F&& f) {
    if constexpr (Start < End) {
        f(std::integral_constant<decltype(Start), Start>());
        ForEach<Start + 1, End>(f);
    }
}

// Copies pixel data from a source to a destionation buffer, performing
// format conversion at the same time
template <vk::Format source_format, vk::Format dest_format, SIMD simd>
constexpr void Convert2(std::span<const u8> source, std::span<u8> dest) {
    constexpr u32 source_components = vk::componentCount(source_format);
    constexpr u32 dest_components = vk::componentCount(dest_format);

    // Query vulkan hpp format traits for the info we need
    constexpr FormatInfo<source_components> source_info{source_format};
    constexpr FormatInfo<dest_components> dest_info{dest_format};

    // Create a table with the required component mapping
    constexpr Mapping<source_info, dest_info> mapping{};

    // Begin conversion
    u32 source_offset = 0;
    u32 dest_offset = 0;
    while (source_offset < source.size()) {
        // Load source pixel
        Pixel<source_info.bytes, simd> source_pixel;
        Pixel<dest_info.bytes, simd> dest_pixel{};

        // Load data into the pixel
        source_pixel.Load(source.data() + source_offset);

        // OPTIMIZATION: Some formats (RGB5A1, A1RGB5) are simply rotations
        // of one another. We can use a faster path for these
        if constexpr (s32 rot = mapping.TestMappingRotation();
                      rot > -1 && mapping.AreBitwiseEqual()) {
            dest_pixel = source_pixel.RotateRight(rot);
        // RGB8 <-> RGBA8 is extrenely common on desktop GPUs
        // so it deserves a special path
        } else if constexpr (true) {
        } else {
            ForEach<0, source_components>([&](auto comp) {
                constexpr u8 dest_comp = (mapping >> (2 * comp)) & 0x3;

                // If the component is not mapped skip it
                if constexpr (dest_comp == 0xFF) {
                    return;
                }

                // Retrieve component
                u32 component = GetComponent<source_format, source_bytes, comp>(source_pixel);

                constexpr bool is_source_float = IsFloat<source_format>(comp);
                constexpr bool is_dest_float = IsFloat<dest_format>(dest_comp);

                // Perform float <-> int conversion (normalization)
                if constexpr (is_source_float && !is_dest_float) {
                    float temp;
                    std::memcpy(&temp, &component, sizeof(float));

                    constexpr u64 mask = (1ull << vk::componentBits(dest_format, dest_comp)) - 1;
                    component = static_cast<u32>(temp * mask);
                } else if constexpr (!is_source_float && is_dest_float) {
                    constexpr u64 mask = (1ull << vk::componentBits(source_format, comp)) - 1;
                    float temp = static_cast<float>(component) / mask;
                    std::memcpy(&component, &temp, sizeof(float));
                }

                SetComponent<dest_format, dest_bytes, dest_comp>(dest_pixel, component);
            });
        }

        // Write destination pixel (dest_bytes includes the padding so we cannot use it here)
        std::memcpy(dest.data() + dest_offset, DataPtr<dest_bytes>(dest_pixel),
                    vk::blockSize(dest_format));

        // Copy next pixel
        source_offset += source_pixel.GetStride();
        dest_offset += dest_pixel.GetStride();
    }
}

// Asign the byte count with an integral type
template <u8 bytes>
struct PackedInt { using type = typename std::array<u8, bytes>; };

template <>
struct PackedInt<1> { using type = u8; };

template <>
struct PackedInt<2> { using type = u16; };

template <>
struct PackedInt<4> { using type = u32; };

template <>
struct PackedInt<8> { using type = u64; };

template <u8 bytes>
using PackedType = typename PackedInt<bytes>::type;

// Returns the pointer to the raw bytes respecting the underlying type
template <u8 bytes>
constexpr u8* DataPtr(PackedType<bytes>& data) {
    if constexpr (std::is_integral_v<PackedType<bytes>>) {
        return reinterpret_cast<u8*>(&data);
    } else {
        return data.data();
    }
}

// Returns true when the specified component is of float type
template <vk::Format format>
constexpr bool IsFloat(u8 component) {
    return std::string_view{vk::componentNumericFormat(format, component)} == "SFLOAT";
}

// Returns the offset in bits of the component from the start of the pixel
template <vk::Format format, u8 component, u8 i = 0>
constexpr u32 GetComponentBitOffset() {
    if constexpr (i == component) {
        return 0;
    } else {
        return vk::componentBits(format, i) +
                GetComponentBitOffset<format, component, i + 1>();
    }
}

// Returns the data located at the specified component
template <vk::Format format, u8 bytes, u8 component>
constexpr u32 GetComponent(PackedType<bytes>& pixel) {
    constexpr u64 bit_offset = GetComponentBitOffset<format, component>();
    constexpr u64 component_bits = vk::componentBits(format, component);
    constexpr u64 mask = (1 << component_bits) - 1;

    // First process packed formats which are easy to extract from
    if constexpr (std::is_integral_v<PackedType<bytes>>) {
        return (pixel >> bit_offset) & mask;
    } else {
        // Assume component_bits and offset are byte aligned. Otherwise
        // this would be extremely complicated
        using ComponentType = PackedType<(component_bits >> 3)>;
        static_assert(component_bits % 8 == 0 && bit_offset % 8 == 0);
        static_assert(std::is_integral_v<ComponentType>);

        constexpr u64 byte_offset = bit_offset >> 3;
        return *reinterpret_cast<ComponentType*>(DataPtr<bytes>(pixel) + byte_offset);
    }
}

template <vk::Format format, u8 bytes, u8 component>
constexpr void SetComponent(PackedType<bytes>& pixel, u32 data) {
    constexpr u64 bit_offset = GetComponentBitOffset<format, component>();
    constexpr u64 component_bits = vk::componentBits(format, component);
    constexpr u64 mask = (1ull << component_bits) - 1;

    // First process packed formats which are easy to write
    if constexpr (std::is_integral_v<PackedType<bytes>>) {
        pixel |= (data & mask) << bit_offset;
    } else {
        // Assume component_bits and offset are byte aligned. Otherwise
        // this would be extremely complicated
        using ComponentType = PackedType<(component_bits >> 3)>;
        static_assert(component_bits % 8 == 0 && bit_offset % 8 == 0);
        static_assert(std::is_integral_v<ComponentType>);

        constexpr u64 byte_offset = bit_offset >> 3;
        *reinterpret_cast<ComponentType*>(DataPtr(pixel) + byte_offset) = data;
    }
}

constexpr bool CanUseRotation();

// Lookup table that maps component i of source format
// to component mapping[i] of the destination format
template <vk::Format source_format, u8 source_components,
          vk::Format dest_format, u8 dest_components>
constexpr auto ComponentMapping() {
    // Since there are at most 4 components we can use 2 bits for each index
    u8 mapping = 0xFF;
    for (u8 i = 0; i < source_components; i++) {
        auto source_name = vk::componentName(source_format, i);
        for (u8 j = 0; j < dest_components; j++) {
            auto dest_name = vk::componentName(dest_format, j);
            if (std::string_view{source_name} == std::string_view{dest_name}) {
                mapping |= ((j & 0x3) << 2 * i);
                break;
            }
        }
    }

    return mapping;
}

// Allows for loop like iteration at compile time
template <auto Start, auto End, class F>
constexpr void ConstexprFor(F&& f) {
    if constexpr (Start < End) {
        f(std::integral_constant<decltype(Start), Start>());
        ConstexprFor<Start + 1, End>(f);
    }
}

// Copies pixel data from a source to a destionation buffer, performing
// format conversion at the same time
template <vk::Format source_format, u8 source_bytes,
          vk::Format dest_format, u8 dest_bytes>
constexpr void Convert(std::span<const u8> source, std::span<u8> dest) {
    constexpr u32 source_components = vk::componentCount(source_format);
    constexpr u32 dest_components = vk::componentCount(dest_format);

    // Create a table with the required component mapping
    constexpr auto mapping = ComponentMapping<source_format, source_components,
                                              dest_format, dest_components>();
    u32 source_offset = 0;
    u32 dest_offset = 0;
    while (source_offset < source.size()) {
        // Load source pixel
        PackedType<source_bytes> source_pixel;
        std::memcpy(DataPtr<source_bytes>(source_pixel),
                    source.data() + source_offset, source_bytes);

        PackedType<dest_bytes> dest_pixel{};

        // OPTIMIZATION: Some formats (RGB5A1, A1RGB5) are simply rotations
        // of one another. We can use a faster path for these

        ConstexprFor<0, source_components>([&](auto comp) {
            constexpr u8 dest_comp = (mapping >> (2 * comp)) & 0x3;

            // If the component is not mapped skip it
            if constexpr (dest_comp == 0xFF) {
                return;
            }

            // Retrieve component
            u32 component = GetComponent<source_format, source_bytes, comp>(source_pixel);

            constexpr bool is_source_float = IsFloat<source_format>(comp);
            constexpr bool is_dest_float = IsFloat<dest_format>(dest_comp);

            // Perform float <-> int conversion (normalization)
            if constexpr (is_source_float && !is_dest_float) {
                float temp;
                std::memcpy(&temp, &component, sizeof(float));

                constexpr u64 mask = (1ull << vk::componentBits(dest_format, dest_comp)) - 1;
                component = static_cast<u32>(temp * mask);
            } else if constexpr (!is_source_float && is_dest_float) {
                constexpr u64 mask = (1ull << vk::componentBits(source_format, comp)) - 1;
                float temp = static_cast<float>(component) / mask;
                std::memcpy(&component, &temp, sizeof(float));
            }

            SetComponent<dest_format, dest_bytes, dest_comp>(dest_pixel, component);
        });

        // Write destination pixel (dest_bytes includes the padding so we cannot use it here)
        std::memcpy(dest.data() + dest_offset, DataPtr<dest_bytes>(dest_pixel),
                    vk::blockSize(dest_format));

        // Copy next pixel
        source_offset += source_bytes;
        dest_offset += dest_bytes;
    }
}

} // namespace VideoCore::Vulkan
