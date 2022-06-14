// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <atomic>
#include <bitset>
#include <cmath>
#include <cstring>
#include <iterator>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>
#include <boost/range/iterator_range.hpp>
#include "common/alignment.h"
#include "common/bit_field.h"
#include "common/color.h"
#include "common/logging/log.h"
#include "common/microprofile.h"
#include "common/scope_exit.h"
#include "common/texture.h"
#include "common/vector_math.h"
#include "core/core.h"
#include "core/frontend/emu_window.h"
#include "core/hle/kernel/process.h"
#include "core/memory.h"
#include "core/settings.h"
#include "video_core/pica_state.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_vulkan/vk_task_scheduler.h"
#include "video_core/renderer_vulkan/vk_rasterizer_cache.h"
#include "video_core/renderer_vulkan/vk_format_reinterpreter.h"
#include "video_core/renderer_vulkan/vk_state.h"
#include "video_core/utils.h"
#include "video_core/video_core.h"

namespace Vulkan {

using SurfaceType = SurfaceParams::SurfaceType;
using PixelFormat = SurfaceParams::PixelFormat;

static constexpr std::array<vk::Format, 5> fb_format_tuples = {{
    vk::Format::eR8G8B8A8Srgb,     // RGBA8
    vk::Format::eR8G8B8Srgb,       // RGB8
    vk::Format::eR5G5B5A1UnormPack16, // RGB5A1
    vk::Format::eR5G6B5UnormPack16,     // RGB565
    vk::Format::eR4G4B4A4UnormPack16,   // RGBA4
}};

static constexpr std::array<vk::Format, 4> depth_format_tuples = {{
    vk::Format::eD16Unorm, // D16
    vk::Format::eUndefined,
    vk::Format::eD24UnormS8Uint,   // D24
    vk::Format::eD24UnormS8Uint, // D24S8
}};

vk::Format GetFormatTuple(PixelFormat pixel_format) {
    const SurfaceType type = SurfaceParams::GetFormatType(pixel_format);
    if (type == SurfaceType::Color) {
        ASSERT(static_cast<std::size_t>(pixel_format) < fb_format_tuples.size());
        return fb_format_tuples[static_cast<unsigned int>(pixel_format)];
    } else if (type == SurfaceType::Depth || type == SurfaceType::DepthStencil) {
        std::size_t tuple_idx = static_cast<std::size_t>(pixel_format) - 14;
        ASSERT(tuple_idx < depth_format_tuples.size());
        return depth_format_tuples[tuple_idx];
    }
    return vk::Format::eR8G8B8A8Srgb;
}

template <typename Map, typename Interval>
static constexpr auto RangeFromInterval(Map& map, const Interval& interval) {
    return boost::make_iterator_range(map.equal_range(interval));
}

template <bool morton_to_gl, PixelFormat format>
static void MortonCopyTile(u32 stride, u8* tile_buffer, u8* gpu_buffer) {
    constexpr u32 bytes_per_pixel = SurfaceParams::GetFormatBpp(format) / 8;
    constexpr u32 vk_bytes_per_pixel = CachedSurface::GetBytesPerPixel(format);
    for (u32 y = 0; y < 8; ++y) {
        for (u32 x = 0; x < 8; ++x) {
            u8* tile_ptr = tile_buffer + VideoCore::MortonInterleave(x, y) * bytes_per_pixel;
            u8* gpu_ptr = gpu_buffer + ((7 - y) * stride + x) * vk_bytes_per_pixel;
            if constexpr (morton_to_gl) {
                if constexpr (format == PixelFormat::D24S8) {
                    gpu_ptr[0] = tile_ptr[3];
                    std::memcpy(gpu_ptr + 1, tile_ptr, 3);
                } else if (format == PixelFormat::RGBA8) {
                    gpu_ptr[0] = tile_ptr[3];
                    gpu_ptr[1] = tile_ptr[2];
                    gpu_ptr[2] = tile_ptr[1];
                    gpu_ptr[3] = tile_ptr[0];
                 } else if (format == PixelFormat::RGB8) {
                     gpu_ptr[0] = tile_ptr[2];
                     gpu_ptr[1] = tile_ptr[1];
                     gpu_ptr[2] = tile_ptr[0];
                 } else {
                     std::memcpy(gpu_ptr, tile_ptr, bytes_per_pixel);
                 }
            } else {
                if constexpr (format == PixelFormat::D24S8) {
                    std::memcpy(tile_ptr, gpu_ptr + 1, 3);
                    tile_ptr[3] = gpu_ptr[0];
                } else if (format == PixelFormat::RGBA8) {
                    // because GLES does not have ABGR format
                    // so we will do byteswapping here
                    tile_ptr[0] = gpu_ptr[3];
                    tile_ptr[1] = gpu_ptr[2];
                    tile_ptr[2] = gpu_ptr[1];
                    tile_ptr[3] = gpu_ptr[0];
                } else if (format == PixelFormat::RGB8) {
                    tile_ptr[0] = gpu_ptr[2];
                    tile_ptr[1] = gpu_ptr[1];
                    tile_ptr[2] = gpu_ptr[0];
                } else {
                    std::memcpy(tile_ptr, gpu_ptr, bytes_per_pixel);
                }
            }
        }
    }
}

template <bool morton_to_gl, PixelFormat format>
static void MortonCopy(u32 stride, u32 height, u8* gpu_buffer, PAddr base, PAddr start, PAddr end) {
    constexpr u32 bytes_per_pixel = SurfaceParams::GetFormatBpp(format) / 8;
    constexpr u32 tile_size = bytes_per_pixel * 64;

    constexpr u32 gl_bytes_per_pixel = CachedSurface::GetBytesPerPixel(format);
    static_assert(gl_bytes_per_pixel >= bytes_per_pixel, "");
    gpu_buffer += gl_bytes_per_pixel - bytes_per_pixel;

    const PAddr aligned_down_start = base + Common::AlignDown(start - base, tile_size);
    const PAddr aligned_start = base + Common::AlignUp(start - base, tile_size);
    const PAddr aligned_end = base + Common::AlignDown(end - base, tile_size);

    ASSERT(!morton_to_gl || (aligned_start == start && aligned_end == end));

    const u32 begin_pixel_index = (aligned_down_start - base) / bytes_per_pixel;
    u32 x = (begin_pixel_index % (stride * 8)) / 8;
    u32 y = (begin_pixel_index / (stride * 8)) * 8;

    gpu_buffer += ((height - 8 - y) * stride + x) * gl_bytes_per_pixel;

    auto gpubuf_next_tile = [&] {
        x = (x + 8) % stride;
        gpu_buffer += 8 * gl_bytes_per_pixel;
        if (!x) {
            y += 8;
            gpu_buffer -= stride * 9 * gl_bytes_per_pixel;
        }
    };

    u8* tile_buffer = VideoCore::g_memory->GetPhysicalPointer(start);

    if (start < aligned_start && !morton_to_gl) {
        std::array<u8, tile_size> tmp_buf;
        MortonCopyTile<morton_to_gl, format>(stride, &tmp_buf[0], gpu_buffer);
        std::memcpy(tile_buffer, &tmp_buf[start - aligned_down_start],
                    std::min(aligned_start, end) - start);

        tile_buffer += aligned_start - start;
        gpubuf_next_tile();
    }

    const u8* const buffer_end = tile_buffer + aligned_end - aligned_start;
    PAddr current_paddr = aligned_start;
    while (tile_buffer < buffer_end) {
        // Pokemon Super Mystery Dungeon will try to use textures that go beyond
        // the end address of VRAM. Stop reading if reaches invalid address
        if (!VideoCore::g_memory->IsValidPhysicalAddress(current_paddr) ||
            !VideoCore::g_memory->IsValidPhysicalAddress(current_paddr + tile_size)) {
            LOG_ERROR(Render_Vulkan, "Out of bound texture");
            break;
        }
        MortonCopyTile<morton_to_gl, format>(stride, tile_buffer, gpu_buffer);
        tile_buffer += tile_size;
        current_paddr += tile_size;
        gpubuf_next_tile();
    }

    if (end > std::max(aligned_start, aligned_end) && !morton_to_gl) {
        std::array<u8, tile_size> tmp_buf;
        MortonCopyTile<morton_to_gl, format>(stride, &tmp_buf[0], gpu_buffer);
        std::memcpy(tile_buffer, &tmp_buf[0], end - aligned_end);
    }
}

static constexpr std::array<void (*)(u32, u32, u8*, PAddr, PAddr, PAddr), 18> morton_to_gpu_fns = {
    MortonCopy<true, PixelFormat::RGBA8>,  // 0
    MortonCopy<true, PixelFormat::RGB8>,   // 1
    MortonCopy<true, PixelFormat::RGB5A1>, // 2
    MortonCopy<true, PixelFormat::RGB565>, // 3
    MortonCopy<true, PixelFormat::RGBA4>,  // 4
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,                             // 5 - 13
    MortonCopy<true, PixelFormat::D16>,  // 14
    nullptr,                             // 15
    MortonCopy<true, PixelFormat::D24>,  // 16
    MortonCopy<true, PixelFormat::D24S8> // 17
};

static constexpr std::array<void (*)(u32, u32, u8*, PAddr, PAddr, PAddr), 18> gpu_to_morton_fns = {
    MortonCopy<false, PixelFormat::RGBA8>,  // 0
    MortonCopy<false, PixelFormat::RGB8>,   // 1
    MortonCopy<false, PixelFormat::RGB5A1>, // 2
    MortonCopy<false, PixelFormat::RGB565>, // 3
    MortonCopy<false, PixelFormat::RGBA4>,  // 4
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,                              // 5 - 13
    MortonCopy<false, PixelFormat::D16>,  // 14
    nullptr,                              // 15
    MortonCopy<false, PixelFormat::D24>,  // 16
    MortonCopy<false, PixelFormat::D24S8> // 17
};

inline vk::ImageSubresourceRange SubResourceLayersToRange(const vk::ImageSubresourceLayers& in) {
    vk::ImageSubresourceRange out;
    out.aspectMask = in.aspectMask;
    out.baseArrayLayer = in.baseArrayLayer;
    out.layerCount = in.layerCount;
    out.baseMipLevel = in.mipLevel;
    out.levelCount = 1;
    return out;
}

static bool BlitTextures(const Surface& src_surface, const Common::Rectangle<u32>& src_rect,
                         const Surface& dst_surface, const Common::Rectangle<u32>& dst_rect, SurfaceType type) {
    vk::ImageSubresourceLayers image_range{{}, {}, 0, 1};
    switch (src_surface->type) {
    case SurfaceParams::SurfaceType::Color:
    case SurfaceParams::SurfaceType::Texture:
        image_range.aspectMask = vk::ImageAspectFlagBits::eColor;
        break;
    case SurfaceParams::SurfaceType::Depth:
        image_range.aspectMask = vk::ImageAspectFlagBits::eDepth;
        break;
    case SurfaceParams::SurfaceType::DepthStencil:
        image_range.aspectMask =
            vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
        break;
    default:
        UNIMPLEMENTED();
    }

    // Prepare images for transfer
    auto cmdbuffer = g_vk_task_scheduler->GetRenderCommandBuffer();

    auto& src_texture = src_surface->texture;
    src_texture.Transition(cmdbuffer, vk::ImageLayout::eTransferSrcOptimal);

    auto& dst_texture = dst_surface->texture;
    dst_texture.Transition(cmdbuffer, vk::ImageLayout::eTransferDstOptimal);

    const std::array src_offsets{
        vk::Offset3D{static_cast<s32>(src_rect.left), static_cast<s32>(src_rect.bottom), 0},
        vk::Offset3D{static_cast<s32>(src_rect.right), static_cast<s32>(src_rect.top), 1}
    };

    const std::array dst_offsets{
        vk::Offset3D{static_cast<s32>(dst_rect.left), static_cast<s32>(dst_rect.bottom), 0},
        vk::Offset3D{static_cast<s32>(dst_rect.right), static_cast<s32>(dst_rect.top), 1}
    };

    vk::ImageBlit blit_area{image_range, src_offsets, image_range, dst_offsets};
    cmdbuffer.blitImage(src_texture.GetHandle(), vk::ImageLayout::eTransferSrcOptimal,
                        dst_texture.GetHandle(), vk::ImageLayout::eTransferDstOptimal,
                        {blit_area}, vk::Filter::eNearest);

    // Revert changes to the layout
    src_texture.Transition(cmdbuffer, vk::ImageLayout::eShaderReadOnlyOptimal);
    dst_texture.Transition(cmdbuffer, vk::ImageLayout::eShaderReadOnlyOptimal);

    return true;
}

static vk::Rect2D FromRect(Common::Rectangle<u32> rect) {
    vk::Offset2D offset{static_cast<s32>(rect.left), static_cast<s32>(rect.bottom)};
    vk::Extent2D extent{rect.GetWidth(), rect.GetHeight()};
    return vk::Rect2D{offset, extent};
}

// Allocate an uninitialized texture of appropriate size and format for the surface
VKTexture RasterizerCacheVulkan::AllocateSurfaceTexture(SurfaceType type, vk::Format format,
                                                        u32 width, u32 height) {
    // First check if the texture can be recycled
    auto recycled_tex = host_texture_recycler.find({format, width, height});
    if (recycled_tex != host_texture_recycler.end()) {
        VKTexture texture = std::move(recycled_tex->second);
        host_texture_recycler.erase(recycled_tex);
        return texture;
    }

    auto GetUsage = [](SurfaceType type) {
        auto usage = vk::ImageUsageFlagBits::eSampled |
                vk::ImageUsageFlagBits::eTransferDst |
                vk::ImageUsageFlagBits::eTransferSrc;

        switch (type) {
        case SurfaceType::Color:
        case SurfaceType::Fill:
        case SurfaceType::Texture:
            usage |= vk::ImageUsageFlagBits::eColorAttachment;
            break;
        case SurfaceType::Depth:
        case SurfaceType::DepthStencil:
            usage |= vk::ImageUsageFlagBits::eDepthStencilAttachment;
            break;
        default:
            break;
        }

        return usage;
    };

    // Otherwise create a brand new texture
    u32 levels = std::log2(std::max(width, height)) + 1;
    VKTexture::Info texture_info = {
        .width = width,
        .height = height,
        .format = format,
        .type = vk::ImageType::e2D,
        .view_type = vk::ImageViewType::e2D,
        .usage = GetUsage(type),
        .levels = levels
    };

    VKTexture texture;
    texture.Create(texture_info);

    auto cmdbuffer = g_vk_task_scheduler->GetUploadCommandBuffer();
    texture.Transition(cmdbuffer, vk::ImageLayout::eShaderReadOnlyOptimal);

    return texture;
}

CachedSurface::~CachedSurface() {
    if (texture.IsValid()) {
        auto tag = is_custom ? HostTextureTag{GetFormatTuple(PixelFormat::RGBA8),
                                              custom_tex_info.width, custom_tex_info.height}
                             : HostTextureTag{GetFormatTuple(pixel_format), GetScaledWidth(),
                                              GetScaledHeight()};

        owner.host_texture_recycler.emplace(tag, std::move(texture));
    }
}

bool CachedSurface::CanFill(const SurfaceParams& dest_surface,
                            SurfaceInterval fill_interval) const {
    if (type == SurfaceType::Fill && IsRegionValid(fill_interval) &&
        boost::icl::first(fill_interval) >= addr &&
        boost::icl::last_next(fill_interval) <= end && // dest_surface is within our fill range
        dest_surface.FromInterval(fill_interval).GetInterval() ==
            fill_interval) { // make sure interval is a rectangle in dest surface
        if (fill_size * 8 != dest_surface.GetFormatBpp()) {
            // Check if bits repeat for our fill_size
            const u32 dest_bytes_per_pixel = std::max(dest_surface.GetFormatBpp() / 8, 1u);
            std::vector<u8> fill_test(fill_size * dest_bytes_per_pixel);

            for (u32 i = 0; i < dest_bytes_per_pixel; ++i)
                std::memcpy(&fill_test[i * fill_size], &fill_data[0], fill_size);

            for (u32 i = 0; i < fill_size; ++i)
                if (std::memcmp(&fill_test[dest_bytes_per_pixel * i], &fill_test[0],
                                dest_bytes_per_pixel) != 0)
                    return false;

            if (dest_surface.GetFormatBpp() == 4 && (fill_test[0] & 0xF) != (fill_test[0] >> 4))
                return false;
        }
        return true;
    }
    return false;
}

bool CachedSurface::CanCopy(const SurfaceParams& dest_surface,
                            SurfaceInterval copy_interval) const {
    SurfaceParams subrect_params = dest_surface.FromInterval(copy_interval);
    ASSERT(subrect_params.GetInterval() == copy_interval);
    if (CanSubRect(subrect_params))
        return true;

    if (CanFill(dest_surface, copy_interval))
        return true;

    return false;
}

MICROPROFILE_DEFINE(Vulkan_CopySurface, "Vulkan", "CopySurface", MP_RGB(128, 192, 64));
void RasterizerCacheVulkan::CopySurface(const Surface& src_surface, const Surface& dst_surface,
                                        SurfaceInterval copy_interval) {
    MICROPROFILE_SCOPE(Vulkan_CopySurface);

    SurfaceParams subrect_params = dst_surface->FromInterval(copy_interval);
    ASSERT(subrect_params.GetInterval() == copy_interval);

    ASSERT(src_surface != dst_surface);

    // This is only called when CanCopy is true, no need to run checks here
    if (src_surface->type == SurfaceType::Fill) {
        // NO-OP Vulkan does not allow easy clearing for arbitary textures with rectangle
        return;
    }
    if (src_surface->CanSubRect(subrect_params)) {
        auto srect = src_surface->GetScaledSubRect(subrect_params);
        auto drect = dst_surface->GetScaledSubRect(subrect_params);

        BlitTextures(src_surface, srect, dst_surface, drect, src_surface->type);
        return;
    }

    UNREACHABLE();
}

MICROPROFILE_DEFINE(Vulkan_SurfaceLoad, "Vulkan", "Surface Load", MP_RGB(128, 192, 64));
void CachedSurface::LoadGPUBuffer(PAddr load_start, PAddr load_end) {
    ASSERT(type != SurfaceType::Fill);

    const u8* const texture_src_data = VideoCore::g_memory->GetPhysicalPointer(addr);
    if (texture_src_data == nullptr)
        return;

    if (vk_buffer.empty()) {
        vk_buffer.resize(width * height * GetBytesPerPixel(pixel_format));
    }

    // TODO: Should probably be done in ::Memory:: and check for other regions too
    if (load_start < Memory::VRAM_VADDR_END && load_end > Memory::VRAM_VADDR_END)
        load_end = Memory::VRAM_VADDR_END;

    if (load_start < Memory::VRAM_VADDR && load_end > Memory::VRAM_VADDR)
        load_start = Memory::VRAM_VADDR;

    MICROPROFILE_SCOPE(Vulkan_SurfaceLoad);

    ASSERT(load_start >= addr && load_end <= end);
    const u32 start_offset = load_start - addr;

    if (!is_tiled) {
        ASSERT(type == SurfaceType::Color);
        std::memcpy(&vk_buffer[start_offset], texture_src_data + start_offset,
                        load_end - load_start);
    } else {
        if (type == SurfaceType::Texture) {
            Pica::Texture::TextureInfo tex_info{};
            tex_info.width = width;
            tex_info.height = height;
            tex_info.format = static_cast<Pica::TexturingRegs::TextureFormat>(pixel_format);
            tex_info.SetDefaultStride();
            tex_info.physical_address = addr;

            const SurfaceInterval load_interval(load_start, load_end);
            const auto rect = GetSubRect(FromInterval(load_interval));
            ASSERT(FromInterval(load_interval).GetInterval() == load_interval);

            for (unsigned y = rect.bottom; y < rect.top; ++y) {
                for (unsigned x = rect.left; x < rect.right; ++x) {
                    auto vec4 =
                        Pica::Texture::LookupTexture(texture_src_data, x, height - 1 - y, tex_info);
                    const std::size_t offset = (x + (width * y)) * 4;
                    std::memcpy(&vk_buffer[offset], vec4.AsArray(), 4);
                }
            }
        } else {
            morton_to_gpu_fns[static_cast<std::size_t>(pixel_format)](stride, height, &vk_buffer[0],
                                                                     addr, load_start, load_end);
        }
    }
}

MICROPROFILE_DEFINE(Vulkan_SurfaceFlush, "Vulkan", "Surface Flush", MP_RGB(128, 192, 64));
void CachedSurface::FlushGPUBuffer(PAddr flush_start, PAddr flush_end) {
    u8* const dst_buffer = VideoCore::g_memory->GetPhysicalPointer(addr);
    if (dst_buffer == nullptr)
        return;

    ASSERT(vk_buffer.size() == width * height * GetBytesPerPixel(pixel_format));

    // TODO: Should probably be done in ::Memory:: and check for other regions too
    // same as loadglbuffer()
    if (flush_start < Memory::VRAM_VADDR_END && flush_end > Memory::VRAM_VADDR_END)
        flush_end = Memory::VRAM_VADDR_END;

    if (flush_start < Memory::VRAM_VADDR && flush_end > Memory::VRAM_VADDR)
        flush_start = Memory::VRAM_VADDR;

    MICROPROFILE_SCOPE(Vulkan_SurfaceFlush);

    ASSERT(flush_start >= addr && flush_end <= end);
    const u32 start_offset = flush_start - addr;
    const u32 end_offset = flush_end - addr;

    if (type == SurfaceType::Fill) {
        const u32 coarse_start_offset = start_offset - (start_offset % fill_size);
        const u32 backup_bytes = start_offset % fill_size;
        std::array<u8, 4> backup_data;
        if (backup_bytes)
            std::memcpy(&backup_data[0], &dst_buffer[coarse_start_offset], backup_bytes);

        for (u32 offset = coarse_start_offset; offset < end_offset; offset += fill_size) {
            std::memcpy(&dst_buffer[offset], &fill_data[0],
                        std::min(fill_size, end_offset - offset));
        }

        if (backup_bytes)
            std::memcpy(&dst_buffer[coarse_start_offset], &backup_data[0], backup_bytes);
    } else if (!is_tiled) {
        if (pixel_format == PixelFormat::RGBA8) {
            for (std::size_t i = start_offset; i < flush_end - addr; i += 4) {
                dst_buffer[i] = vk_buffer[i + 3];
                dst_buffer[i + 1] = vk_buffer[i + 2];
                dst_buffer[i + 2] = vk_buffer[i + 1];
                dst_buffer[i + 3] = vk_buffer[i];
            }
        } else if (pixel_format == PixelFormat::RGB8) {
            for (std::size_t i = start_offset; i < flush_end - addr; i += 3) {
                dst_buffer[i] = vk_buffer[i + 2];
                dst_buffer[i + 1] = vk_buffer[i + 1];
                dst_buffer[i + 2] = vk_buffer[i];
            }
        } else {
            std::memcpy(dst_buffer + start_offset, &vk_buffer[start_offset],
                        flush_end - flush_start);
        }
    } else {
        gpu_to_morton_fns[static_cast<std::size_t>(pixel_format)](stride, height, &vk_buffer[0],
                                                                 addr, flush_start, flush_end);
    }
}

MICROPROFILE_DEFINE(Vulkan_TextureUL, "Vulkan", "Texture Upload", MP_RGB(128, 192, 64));
void CachedSurface::UploadGPUTexture(Common::Rectangle<u32> rect) {
    if (type == SurfaceType::Fill)
        return;

    MICROPROFILE_SCOPE(Vulkan_TextureUL);

    ASSERT(vk_buffer.size() == width * height * GetBytesPerPixel(pixel_format));

    // TODO: Handle resolution scaling and custom textures

    // Load data from memory to the surface
    auto buffer_offset = (rect.bottom * stride + rect.left) * GetBytesPerPixel(pixel_format);
    auto update_size = rect.GetWidth() * rect.GetHeight() * GetBytesPerPixel(pixel_format);
    std::span<u8> memory(vk_buffer.data() + buffer_offset, update_size);

    texture.Upload(0, 0, stride, FromRect(rect), memory);

    InvalidateAllWatcher();
}

MICROPROFILE_DEFINE(OpenGL_TextureDL, "OpenGL", "Texture Download", MP_RGB(128, 192, 64));
void CachedSurface::DownloadGPUTexture(const Common::Rectangle<u32>& rect) {
    if (type == SurfaceType::Fill) {
        return;
    }

    MICROPROFILE_SCOPE(OpenGL_TextureDL);

    if (vk_buffer.empty()) {
        vk_buffer.resize(width * height * GetBytesPerPixel(pixel_format));
    }

    // TODO: Handle resolution scaling and custom textures

    auto buffer_offset = (rect.bottom * stride + rect.left) * GetBytesPerPixel(pixel_format);
    auto buffer_size = rect.GetWidth() * rect.GetHeight() * GetBytesPerPixel(pixel_format);
    std::span<u8> memory(vk_buffer.data() + buffer_offset, buffer_size);

    texture.Download(0, 0, stride, FromRect(rect), memory);
}

enum MatchFlags {
    Invalid = 1,      // Flag that can be applied to other match types, invalid matches require
                      // validation before they can be used
    Exact = 1 << 1,   // Surfaces perfectly match
    SubRect = 1 << 2, // Surface encompasses params
    Copy = 1 << 3,    // Surface we can copy from
    Expand = 1 << 4,  // Surface that can expand params
    TexCopy = 1 << 5  // Surface that will match a display transfer "texture copy" parameters
};

static constexpr MatchFlags operator|(MatchFlags lhs, MatchFlags rhs) {
    return static_cast<MatchFlags>(static_cast<int>(lhs) | static_cast<int>(rhs));
}

/// Get the best surface match (and its match type) for the given flags
template <MatchFlags find_flags>
static Surface FindMatch(const SurfaceCache& surface_cache, const SurfaceParams& params,
                         ScaleMatch match_scale_type,
                         std::optional<SurfaceInterval> validate_interval = std::nullopt) {
    Surface match_surface = nullptr;
    bool match_valid = false;
    u32 match_scale = 0;
    SurfaceInterval match_interval{};

    for (const auto& pair : RangeFromInterval(surface_cache, params.GetInterval())) {
        for (const auto& surface : pair.second) {
            const bool res_scale_matched = match_scale_type == ScaleMatch::Exact
                                               ? (params.res_scale == surface->res_scale)
                                               : (params.res_scale <= surface->res_scale);
            // validity will be checked in GetCopyableInterval
            bool is_valid =
                find_flags & MatchFlags::Copy
                    ? true
                    : surface->IsRegionValid(validate_interval.value_or(params.GetInterval()));

            if (!(find_flags & MatchFlags::Invalid) && !is_valid)
                continue;

            auto IsMatch_Helper = [&](auto check_type, auto match_fn) {
                if (!(find_flags & check_type))
                    return;

                bool matched;
                SurfaceInterval surface_interval;
                std::tie(matched, surface_interval) = match_fn();
                if (!matched)
                    return;

                if (!res_scale_matched && match_scale_type != ScaleMatch::Ignore &&
                    surface->type != SurfaceType::Fill)
                    return;

                // Found a match, update only if this is better than the previous one
                auto UpdateMatch = [&] {
                    match_surface = surface;
                    match_valid = is_valid;
                    match_scale = surface->res_scale;
                    match_interval = surface_interval;
                };

                if (surface->res_scale > match_scale) {
                    UpdateMatch();
                    return;
                } else if (surface->res_scale < match_scale) {
                    return;
                }

                if (is_valid && !match_valid) {
                    UpdateMatch();
                    return;
                } else if (is_valid != match_valid) {
                    return;
                }

                if (boost::icl::length(surface_interval) > boost::icl::length(match_interval)) {
                    UpdateMatch();
                }
            };
            IsMatch_Helper(std::integral_constant<MatchFlags, MatchFlags::Exact>{}, [&] {
                return std::make_pair(surface->ExactMatch(params), surface->GetInterval());
            });
            IsMatch_Helper(std::integral_constant<MatchFlags, MatchFlags::SubRect>{}, [&] {
                return std::make_pair(surface->CanSubRect(params), surface->GetInterval());
            });
            IsMatch_Helper(std::integral_constant<MatchFlags, MatchFlags::Copy>{}, [&] {
                ASSERT(validate_interval);
                auto copy_interval =
                    params.FromInterval(*validate_interval).GetCopyableInterval(surface);
                bool matched = boost::icl::length(copy_interval & *validate_interval) != 0 &&
                               surface->CanCopy(params, copy_interval);
                return std::make_pair(matched, copy_interval);
            });
            IsMatch_Helper(std::integral_constant<MatchFlags, MatchFlags::Expand>{}, [&] {
                return std::make_pair(surface->CanExpand(params), surface->GetInterval());
            });
            IsMatch_Helper(std::integral_constant<MatchFlags, MatchFlags::TexCopy>{}, [&] {
                return std::make_pair(surface->CanTexCopy(params), surface->GetInterval());
            });
        }
    }
    return match_surface;
}

RasterizerCacheVulkan::RasterizerCacheVulkan() {
    resolution_scale_factor = VideoCore::GetResolutionScaleFactor();
    //format_reinterpreter = std::make_unique<FormatReinterpreterVulkan>();
}

RasterizerCacheVulkan::~RasterizerCacheVulkan() {
#ifndef ANDROID
    // This is for switching renderers, which is unsupported on Android, and costly on shutdown
    ClearAll(false);
#endif
}

MICROPROFILE_DEFINE(OpenGL_BlitSurface, "OpenGL", "BlitSurface", MP_RGB(128, 192, 64));
bool RasterizerCacheVulkan::BlitSurfaces(const Surface& src_surface,
                                         const Common::Rectangle<u32>& src_rect,
                                         const Surface& dst_surface,
                                         const Common::Rectangle<u32>& dst_rect) {
    MICROPROFILE_SCOPE(OpenGL_BlitSurface);

    if (!SurfaceParams::CheckFormatsBlittable(src_surface->pixel_format, dst_surface->pixel_format))
        return false;

    dst_surface->InvalidateAllWatcher();
    return BlitTextures(src_surface, src_rect, dst_surface, dst_rect, src_surface->type);
}

Surface RasterizerCacheVulkan::GetSurface(const SurfaceParams& params, ScaleMatch match_res_scale,
                                          bool load_if_create) {
    if (params.addr == 0 || params.height * params.width == 0) {
        return nullptr;
    }
    // Use GetSurfaceSubRect instead
    ASSERT(params.width == params.stride);

    ASSERT(!params.is_tiled || (params.width % 8 == 0 && params.height % 8 == 0));

    // Check for an exact match in existing surfaces
    Surface surface =
        FindMatch<MatchFlags::Exact | MatchFlags::Invalid>(surface_cache, params, match_res_scale);

    if (surface == nullptr) {
        u16 target_res_scale = params.res_scale;
        if (match_res_scale != ScaleMatch::Exact) {
            // This surface may have a subrect of another surface with a higher res_scale, find
            // it to adjust our params
            SurfaceParams find_params = params;
            Surface expandable = FindMatch<MatchFlags::Expand | MatchFlags::Invalid>(
                surface_cache, find_params, match_res_scale);
            if (expandable != nullptr && expandable->res_scale > target_res_scale) {
                target_res_scale = expandable->res_scale;
            }
            // Keep res_scale when reinterpreting d24s8 -> rgba8
            if (params.pixel_format == PixelFormat::RGBA8) {
                find_params.pixel_format = PixelFormat::D24S8;
                expandable = FindMatch<MatchFlags::Expand | MatchFlags::Invalid>(
                    surface_cache, find_params, match_res_scale);
                if (expandable != nullptr && expandable->res_scale > target_res_scale) {
                    target_res_scale = expandable->res_scale;
                }
            }
        }
        SurfaceParams new_params = params;
        new_params.res_scale = target_res_scale;
        surface = CreateSurface(new_params);
        RegisterSurface(surface);
    }

    if (load_if_create) {
        ValidateSurface(surface, params.addr, params.size);
    }

    return surface;
}

SurfaceRect_Tuple RasterizerCacheVulkan::GetSurfaceSubRect(const SurfaceParams& params,
                                                           ScaleMatch match_res_scale,
                                                           bool load_if_create) {
    if (params.addr == 0 || params.height * params.width == 0) {
        return std::make_tuple(nullptr, Common::Rectangle<u32>{});
    }

    // Attempt to find encompassing surface
    Surface surface = FindMatch<MatchFlags::SubRect | MatchFlags::Invalid>(surface_cache, params,
                                                                           match_res_scale);

    // Check if FindMatch failed because of res scaling
    // If that's the case create a new surface with
    // the dimensions of the lower res_scale surface
    // to suggest it should not be used again
    if (surface == nullptr && match_res_scale != ScaleMatch::Ignore) {
        surface = FindMatch<MatchFlags::SubRect | MatchFlags::Invalid>(surface_cache, params,
                                                                       ScaleMatch::Ignore);
        if (surface != nullptr) {
            SurfaceParams new_params = *surface;
            new_params.res_scale = params.res_scale;

            surface = CreateSurface(new_params);
            RegisterSurface(surface);
        }
    }

    SurfaceParams aligned_params = params;
    if (params.is_tiled) {
        aligned_params.height = Common::AlignUp(params.height, 8);
        aligned_params.width = Common::AlignUp(params.width, 8);
        aligned_params.stride = Common::AlignUp(params.stride, 8);
        aligned_params.UpdateParams();
    }

    // Check for a surface we can expand before creating a new one
    if (surface == nullptr) {
        surface = FindMatch<MatchFlags::Expand | MatchFlags::Invalid>(surface_cache, aligned_params,
                                                                      match_res_scale);
        if (surface != nullptr) {
            aligned_params.width = aligned_params.stride;
            aligned_params.UpdateParams();

            SurfaceParams new_params = *surface;
            new_params.addr = std::min(aligned_params.addr, surface->addr);
            new_params.end = std::max(aligned_params.end, surface->end);
            new_params.size = new_params.end - new_params.addr;
            new_params.height =
                new_params.size / aligned_params.BytesInPixels(aligned_params.stride);
            ASSERT(new_params.size % aligned_params.BytesInPixels(aligned_params.stride) == 0);

            Surface new_surface = CreateSurface(new_params);
            DuplicateSurface(surface, new_surface);

            // Delete the expanded surface, this can't be done safely yet
            // because it may still be in use
            surface->UnlinkAllWatcher(); // unlink watchers as if this surface is already deleted
            remove_surfaces.emplace(surface);

            surface = new_surface;
            RegisterSurface(new_surface);
        }
    }

    // No subrect found - create and return a new surface
    if (surface == nullptr) {
        SurfaceParams new_params = aligned_params;
        // Can't have gaps in a surface
        new_params.width = aligned_params.stride;
        new_params.UpdateParams();
        // GetSurface will create the new surface and possibly adjust res_scale if necessary
        surface = GetSurface(new_params, match_res_scale, load_if_create);
    } else if (load_if_create) {
        ValidateSurface(surface, aligned_params.addr, aligned_params.size);
    }

    return std::make_tuple(surface, surface->GetScaledSubRect(params));
}

Surface RasterizerCacheVulkan::GetTextureSurface(
    const Pica::TexturingRegs::FullTextureConfig& config) {
    Pica::Texture::TextureInfo info =
        Pica::Texture::TextureInfo::FromPicaRegister(config.config, config.format);
    return GetTextureSurface(info, config.config.lod.max_level);
}

Surface RasterizerCacheVulkan::GetTextureSurface(const Pica::Texture::TextureInfo& info,
                                                 u32 max_level) {
    if (info.physical_address == 0) {
        return nullptr;
    }

    SurfaceParams params;
    params.addr = info.physical_address;
    params.width = info.width;
    params.height = info.height;
    params.is_tiled = true;
    params.pixel_format = SurfaceParams::PixelFormatFromTextureFormat(info.format);
    //params.res_scale = texture_filterer->IsNull() ? 1 : resolution_scale_factor;
    params.res_scale = 1;
    params.UpdateParams();

    u32 min_width = info.width >> max_level;
    u32 min_height = info.height >> max_level;
    if (min_width % 8 != 0 || min_height % 8 != 0) {
        LOG_CRITICAL(Render_Vulkan, "Texture size ({}x{}) is not multiple of 8", min_width,
                     min_height);
        return nullptr;
    }
    if (info.width != (min_width << max_level) || info.height != (min_height << max_level)) {
        LOG_CRITICAL(Render_Vulkan,
                     "Texture size ({}x{}) does not support required mipmap level ({})",
                     params.width, params.height, max_level);
        return nullptr;
    }

    auto surface = GetSurface(params, ScaleMatch::Ignore, true);
    if (!surface)
        return nullptr;

    // Update mipmap if necessary
    if (max_level != 0) {
        if (max_level >= 8) {
            // since PICA only supports texture size between 8 and 1024, there are at most eight
            // possible mipmap levels including the base.
            LOG_CRITICAL(Render_Vulkan, "Unsupported mipmap level {}", max_level);
            return nullptr;
        }

        SurfaceParams surface_params = *surface;
        for (u32 level = 1; level <= max_level; ++level) {
            // In PICA all mipmap levels are stored next to each other
            surface_params.addr +=
                surface_params.width * surface_params.height * surface_params.GetFormatBpp() / 8;
            surface_params.width /= 2;
            surface_params.height /= 2;
            surface_params.stride = 0; // reset stride and let UpdateParams re-initialize it
            surface_params.UpdateParams();
            auto& watcher = surface->level_watchers[level - 1];
            if (!watcher || !watcher->Get()) {
                auto level_surface = GetSurface(surface_params, ScaleMatch::Ignore, true);
                if (level_surface) {
                    watcher = level_surface->CreateWatcher();
                } else {
                    watcher = nullptr;
                }
            }

            if (watcher && !watcher->IsValid()) {
                auto level_surface = watcher->Get();
                if (!level_surface->invalid_regions.empty()) {
                    ValidateSurface(level_surface, level_surface->addr, level_surface->size);
                }

                //state.ResetTexture(level_surface->texture.handle);
                //state.Apply();
                if (!surface->is_custom /*&& texture_filterer->IsNull()*/) {
                    auto src_rect = level_surface->GetScaledRect();
                    auto dst_rect = surface_params.GetScaledRect();
                    BlitSurfaces(level_surface, src_rect, surface, dst_rect);
                }
                watcher->Validate();
            }
        }
    }

    return surface;
}

SurfaceSurfaceRect_Tuple RasterizerCacheVulkan::GetFramebufferSurfaces(
    bool using_color_fb, bool using_depth_fb, const Common::Rectangle<s32>& viewport_rect) {
    const auto& regs = Pica::g_state.regs;
    const auto& config = regs.framebuffer.framebuffer;

    // update resolution_scale_factor and reset cache if changed
    /*if ((resolution_scale_factor != VideoCore::GetResolutionScaleFactor()) |
        (VideoCore::g_texture_filter_update_requested.exchange(false) &&
         texture_filterer->Reset(Settings::values.texture_filter_name, resolution_scale_factor))) {
        resolution_scale_factor = VideoCore::GetResolutionScaleFactor();
        FlushAll();
        while (!surface_cache.empty())
            UnregisterSurface(*surface_cache.begin()->second.begin());
        texture_cube_cache.clear();
    }*/

    Common::Rectangle<u32> viewport_clamped{
        static_cast<u32>(std::clamp(viewport_rect.left, 0, static_cast<s32>(config.GetWidth()))),
        static_cast<u32>(std::clamp(viewport_rect.top, 0, static_cast<s32>(config.GetHeight()))),
        static_cast<u32>(std::clamp(viewport_rect.right, 0, static_cast<s32>(config.GetWidth()))),
        static_cast<u32>(
            std::clamp(viewport_rect.bottom, 0, static_cast<s32>(config.GetHeight())))};

    // get color and depth surfaces
    SurfaceParams color_params;
    color_params.is_tiled = true;
    color_params.res_scale = resolution_scale_factor;
    color_params.width = config.GetWidth();
    color_params.height = config.GetHeight();
    SurfaceParams depth_params = color_params;

    color_params.addr = config.GetColorBufferPhysicalAddress();
    color_params.pixel_format = SurfaceParams::PixelFormatFromColorFormat(config.color_format);
    color_params.UpdateParams();

    depth_params.addr = config.GetDepthBufferPhysicalAddress();
    depth_params.pixel_format = SurfaceParams::PixelFormatFromDepthFormat(config.depth_format);
    depth_params.UpdateParams();

    auto color_vp_interval = color_params.GetSubRectInterval(viewport_clamped);
    auto depth_vp_interval = depth_params.GetSubRectInterval(viewport_clamped);

    // Make sure that framebuffers don't overlap if both color and depth are being used
    if (using_color_fb && using_depth_fb &&
        boost::icl::length(color_vp_interval & depth_vp_interval)) {
        LOG_CRITICAL(Render_OpenGL, "Color and depth framebuffer memory regions overlap; "
                                    "overlapping framebuffers not supported!");
        using_depth_fb = false;
    }

    Common::Rectangle<u32> color_rect{};
    Surface color_surface = nullptr;
    if (using_color_fb)
        std::tie(color_surface, color_rect) =
            GetSurfaceSubRect(color_params, ScaleMatch::Exact, false);

    Common::Rectangle<u32> depth_rect{};
    Surface depth_surface = nullptr;
    if (using_depth_fb)
        std::tie(depth_surface, depth_rect) =
            GetSurfaceSubRect(depth_params, ScaleMatch::Exact, false);

    Common::Rectangle<u32> fb_rect{};
    if (color_surface != nullptr && depth_surface != nullptr) {
        fb_rect = color_rect;
        // Color and Depth surfaces must have the same dimensions and offsets
        if (color_rect.bottom != depth_rect.bottom || color_rect.top != depth_rect.top ||
            color_rect.left != depth_rect.left || color_rect.right != depth_rect.right) {
            color_surface = GetSurface(color_params, ScaleMatch::Exact, false);
            depth_surface = GetSurface(depth_params, ScaleMatch::Exact, false);
            fb_rect = color_surface->GetScaledRect();
        }
    } else if (color_surface != nullptr) {
        fb_rect = color_rect;
    } else if (depth_surface != nullptr) {
        fb_rect = depth_rect;
    }

    if (color_surface != nullptr) {
        ValidateSurface(color_surface, boost::icl::first(color_vp_interval),
                        boost::icl::length(color_vp_interval));
        color_surface->InvalidateAllWatcher();
    }
    if (depth_surface != nullptr) {
        ValidateSurface(depth_surface, boost::icl::first(depth_vp_interval),
                        boost::icl::length(depth_vp_interval));
        depth_surface->InvalidateAllWatcher();
    }

    return std::make_tuple(color_surface, depth_surface, fb_rect);
}

Surface RasterizerCacheVulkan::GetFillSurface(const GPU::Regs::MemoryFillConfig& config) {
    Surface new_surface = std::make_shared<CachedSurface>(*this);

    new_surface->addr = config.GetStartAddress();
    new_surface->end = config.GetEndAddress();
    new_surface->size = new_surface->end - new_surface->addr;
    new_surface->type = SurfaceType::Fill;
    new_surface->res_scale = std::numeric_limits<u16>::max();

    std::memcpy(&new_surface->fill_data[0], &config.value_32bit, 4);
    if (config.fill_32bit) {
        new_surface->fill_size = 4;
    } else if (config.fill_24bit) {
        new_surface->fill_size = 3;
    } else {
        new_surface->fill_size = 2;
    }

    RegisterSurface(new_surface);
    return new_surface;
}

SurfaceRect_Tuple RasterizerCacheVulkan::GetTexCopySurface(const SurfaceParams& params) {
    Common::Rectangle<u32> rect{};

    Surface match_surface = FindMatch<MatchFlags::TexCopy | MatchFlags::Invalid>(
        surface_cache, params, ScaleMatch::Ignore);

    if (match_surface != nullptr) {
        ValidateSurface(match_surface, params.addr, params.size);

        SurfaceParams match_subrect;
        if (params.width != params.stride) {
            const u32 tiled_size = match_surface->is_tiled ? 8 : 1;
            match_subrect = params;
            match_subrect.width = match_surface->PixelsInBytes(params.width) / tiled_size;
            match_subrect.stride = match_surface->PixelsInBytes(params.stride) / tiled_size;
            match_subrect.height *= tiled_size;
        } else {
            match_subrect = match_surface->FromInterval(params.GetInterval());
            ASSERT(match_subrect.GetInterval() == params.GetInterval());
        }

        rect = match_surface->GetScaledSubRect(match_subrect);
    }

    return std::make_tuple(match_surface, rect);
}

void RasterizerCacheVulkan::DuplicateSurface(const Surface& src_surface,
                                             const Surface& dest_surface) {
    ASSERT(dest_surface->addr <= src_surface->addr && dest_surface->end >= src_surface->end);

    BlitSurfaces(src_surface, src_surface->GetScaledRect(), dest_surface,
                 dest_surface->GetScaledSubRect(*src_surface));

    dest_surface->invalid_regions -= src_surface->GetInterval();
    dest_surface->invalid_regions += src_surface->invalid_regions;

    SurfaceRegions regions;
    for (const auto& pair : RangeFromInterval(dirty_regions, src_surface->GetInterval())) {
        if (pair.second == src_surface) {
            regions += pair.first;
        }
    }
    for (const auto& interval : regions) {
        dirty_regions.set({interval, dest_surface});
    }
}

void RasterizerCacheVulkan::ValidateSurface(const Surface& surface, PAddr addr, u32 size) {
    if (size == 0)
        return;

    const SurfaceInterval validate_interval(addr, addr + size);

    if (surface->type == SurfaceType::Fill) {
        // Sanity check, fill surfaces will always be valid when used
        ASSERT(surface->IsRegionValid(validate_interval));
        return;
    }

    auto validate_regions = surface->invalid_regions & validate_interval;
    auto notify_validated = [&](SurfaceInterval interval) {
        surface->invalid_regions.erase(interval);
        validate_regions.erase(interval);
    };

    while (true) {
        const auto it = validate_regions.begin();
        if (it == validate_regions.end())
            break;

        const auto interval = *it & validate_interval;
        // Look for a valid surface to copy from
        SurfaceParams params = surface->FromInterval(interval);

        Surface copy_surface =
            FindMatch<MatchFlags::Copy>(surface_cache, params, ScaleMatch::Ignore, interval);
        if (copy_surface != nullptr) {
            SurfaceInterval copy_interval = params.GetCopyableInterval(copy_surface);
            CopySurface(copy_surface, surface, copy_interval);
            notify_validated(copy_interval);
            continue;
        }

        // Try to find surface in cache with different format
        // that can can be reinterpreted to the requested format.
        if (ValidateByReinterpretation(surface, params, interval)) {
            notify_validated(interval);
            continue;
        }
        // Could not find a matching reinterpreter, check if we need to implement a
        // reinterpreter
        if (NoUnimplementedReinterpretations(surface, params, interval) &&
            !IntervalHasInvalidPixelFormat(params, interval)) {
            // No surfaces were found in the cache that had a matching bit-width.
            // If the region was created entirely on the GPU,
            // assume it was a developer mistake and skip flushing.
            if (boost::icl::contains(dirty_regions, interval)) {
                LOG_DEBUG(Render_OpenGL, "Region created fully on GPU and reinterpretation is "
                                         "invalid. Skipping validation");
                validate_regions.erase(interval);
                continue;
            }
        }

        // Load data from 3DS memory
        FlushRegion(params.addr, params.size);
        surface->LoadGPUBuffer(params.addr, params.end);
        surface->UploadGPUTexture(surface->GetSubRect(params));
        notify_validated(params.GetInterval());
    }
}

bool RasterizerCacheVulkan::NoUnimplementedReinterpretations(const Surface& surface,
                                                             SurfaceParams& params,
                                                             const SurfaceInterval& interval) {
    static constexpr std::array<PixelFormat, 17> all_formats{
        PixelFormat::RGBA8, PixelFormat::RGB8,   PixelFormat::RGB5A1, PixelFormat::RGB565,
        PixelFormat::RGBA4, PixelFormat::IA8,    PixelFormat::RG8,    PixelFormat::I8,
        PixelFormat::A8,    PixelFormat::IA4,    PixelFormat::I4,     PixelFormat::A4,
        PixelFormat::ETC1,  PixelFormat::ETC1A4, PixelFormat::D16,    PixelFormat::D24,
        PixelFormat::D24S8,
    };
    bool implemented = true;
    for (PixelFormat format : all_formats) {
        if (SurfaceParams::GetFormatBpp(format) == surface->GetFormatBpp()) {
            params.pixel_format = format;
            // This could potentially be expensive,
            // although experimentally it hasn't been too bad
            Surface test_surface =
                FindMatch<MatchFlags::Copy>(surface_cache, params, ScaleMatch::Ignore, interval);
            if (test_surface != nullptr) {
                LOG_WARNING(Render_OpenGL, "Missing pixel_format reinterpreter: {} -> {}",
                            SurfaceParams::PixelFormatAsString(format),
                            SurfaceParams::PixelFormatAsString(surface->pixel_format));
                implemented = false;
            }
        }
    }
    return implemented;
}

bool RasterizerCacheVulkan::IntervalHasInvalidPixelFormat(SurfaceParams& params,
                                                          const SurfaceInterval& interval) {
    params.pixel_format = PixelFormat::Invalid;
    for (const auto& set : RangeFromInterval(surface_cache, interval))
        for (const auto& surface : set.second)
            if (surface->pixel_format == PixelFormat::Invalid) {
                LOG_WARNING(Render_OpenGL, "Surface found with invalid pixel format");
                return true;
            }
    return false;
}

bool RasterizerCacheVulkan::ValidateByReinterpretation(const Surface& surface,
                                                       SurfaceParams& params,
                                                       const SurfaceInterval& interval) {
    /*auto [cvt_begin, cvt_end] =
        format_reinterpreter->GetPossibleReinterpretations(surface->pixel_format);
    for (auto reinterpreter = cvt_begin; reinterpreter != cvt_end; ++reinterpreter) {
        PixelFormat format = reinterpreter->first.src_format;
        params.pixel_format = format;
        Surface reinterpret_surface =
            FindMatch<MatchFlags::Copy>(surface_cache, params, ScaleMatch::Ignore, interval);

        if (reinterpret_surface != nullptr) {
            SurfaceInterval reinterpret_interval = params.GetCopyableInterval(reinterpret_surface);
            SurfaceParams reinterpret_params = surface->FromInterval(reinterpret_interval);
            auto src_rect = reinterpret_surface->GetScaledSubRect(reinterpret_params);
            auto dest_rect = surface->GetScaledSubRect(reinterpret_params);

            reinterpreter->second->Reinterpret(reinterpret_surface, src_rect, surface, dest_rect);
            return true;
        }
    }*/
    return false;
}

void RasterizerCacheVulkan::ClearAll(bool flush) {
    const auto flush_interval = PageMap::interval_type::right_open(0x0, 0xFFFFFFFF);
    // Force flush all surfaces from the cache
    if (flush) {
        FlushRegion(0x0, 0xFFFFFFFF);
    }
    // Unmark all of the marked pages
    for (auto& pair : RangeFromInterval(cached_pages, flush_interval)) {
        const auto interval = pair.first & flush_interval;

        const PAddr interval_start_addr = boost::icl::first(interval) << Memory::PAGE_BITS;
        const PAddr interval_end_addr = boost::icl::last_next(interval) << Memory::PAGE_BITS;
        const u32 interval_size = interval_end_addr - interval_start_addr;

        VideoCore::g_memory->RasterizerMarkRegionCached(interval_start_addr, interval_size, false);
    }

    // Remove the whole cache without really looking at it.
    cached_pages -= flush_interval;
    dirty_regions -= SurfaceInterval(0x0, 0xFFFFFFFF);
    surface_cache -= SurfaceInterval(0x0, 0xFFFFFFFF);
    remove_surfaces.clear();
}

void RasterizerCacheVulkan::FlushRegion(PAddr addr, u32 size, Surface flush_surface) {
    std::lock_guard lock{mutex};

    if (size == 0)
        return;

    const SurfaceInterval flush_interval(addr, addr + size);
    SurfaceRegions flushed_intervals;

    for (auto& pair : RangeFromInterval(dirty_regions, flush_interval)) {
        // small sizes imply that this most likely comes from the cpu, flush the entire region
        // the point is to avoid thousands of small writes every frame if the cpu decides to
        // access that region, anything higher than 8 you're guaranteed it comes from a service
        const auto interval = size <= 8 ? pair.first : pair.first & flush_interval;
        auto& surface = pair.second;

        if (flush_surface != nullptr && surface != flush_surface)
            continue;

        // Sanity check, this surface is the last one that marked this region dirty
        ASSERT(surface->IsRegionValid(interval));

        if (surface->type != SurfaceType::Fill) {
            SurfaceParams params = surface->FromInterval(interval);
            surface->DownloadGPUTexture(surface->GetSubRect(params));
        }

        surface->FlushGPUBuffer(boost::icl::first(interval), boost::icl::last_next(interval));
        flushed_intervals += interval;
    }
    // Reset dirty regions
    dirty_regions -= flushed_intervals;
}

void RasterizerCacheVulkan::FlushAll() {
    FlushRegion(0, 0xFFFFFFFF);
}

void RasterizerCacheVulkan::InvalidateRegion(PAddr addr, u32 size, const Surface& region_owner) {
    std::lock_guard lock{mutex};

    if (size == 0)
        return;

    const SurfaceInterval invalid_interval(addr, addr + size);

    if (region_owner != nullptr) {
        ASSERT(region_owner->type != SurfaceType::Texture);
        ASSERT(addr >= region_owner->addr && addr + size <= region_owner->end);
        // Surfaces can't have a gap
        ASSERT(region_owner->width == region_owner->stride);
        region_owner->invalid_regions.erase(invalid_interval);
    }

    for (const auto& pair : RangeFromInterval(surface_cache, invalid_interval)) {
        for (const auto& cached_surface : pair.second) {
            if (cached_surface == region_owner)
                continue;

            // If cpu is invalidating this region we want to remove it
            // to (likely) mark the memory pages as uncached
            if (region_owner == nullptr && size <= 8) {
                FlushRegion(cached_surface->addr, cached_surface->size, cached_surface);
                remove_surfaces.emplace(cached_surface);
                continue;
            }

            const auto interval = cached_surface->GetInterval() & invalid_interval;
            cached_surface->invalid_regions.insert(interval);
            cached_surface->InvalidateAllWatcher();

            // If the surface has no salvageable data it should be removed from the cache to avoid
            // clogging the data structure
            if (cached_surface->IsSurfaceFullyInvalid()) {
                remove_surfaces.emplace(cached_surface);
            }
        }
    }

    if (region_owner != nullptr)
        dirty_regions.set({invalid_interval, region_owner});
    else
        dirty_regions.erase(invalid_interval);

    for (const auto& remove_surface : remove_surfaces) {
        if (remove_surface == region_owner) {
            Surface expanded_surface = FindMatch<MatchFlags::SubRect | MatchFlags::Invalid>(
                surface_cache, *region_owner, ScaleMatch::Ignore);
            ASSERT(expanded_surface);

            if ((region_owner->invalid_regions - expanded_surface->invalid_regions).empty()) {
                DuplicateSurface(region_owner, expanded_surface);
            } else {
                continue;
            }
        }
        UnregisterSurface(remove_surface);
    }

    remove_surfaces.clear();
}

Surface RasterizerCacheVulkan::CreateSurface(const SurfaceParams& params) {
    Surface surface = std::make_shared<CachedSurface>(*this);
    static_cast<SurfaceParams&>(*surface) = params;

    surface->invalid_regions.insert(surface->GetInterval());
    surface->texture = AllocateSurfaceTexture(params.type, GetFormatTuple(surface->pixel_format),
                                              surface->GetScaledWidth(), surface->GetScaledHeight());
    return surface;
}

void RasterizerCacheVulkan::RegisterSurface(const Surface& surface) {
    std::lock_guard lock{mutex};

    if (surface->registered) {
        return;
    }
    surface->registered = true;
    surface_cache.add({surface->GetInterval(), SurfaceSet{surface}});
    UpdatePagesCachedCount(surface->addr, surface->size, 1);
}

void RasterizerCacheVulkan::UnregisterSurface(const Surface& surface) {
    std::lock_guard lock{mutex};

    if (!surface->registered) {
        return;
    }
    surface->registered = false;
    UpdatePagesCachedCount(surface->addr, surface->size, -1);
    surface_cache.subtract({surface->GetInterval(), SurfaceSet{surface}});
}

void RasterizerCacheVulkan::UpdatePagesCachedCount(PAddr addr, u32 size, int delta) {
    const u32 num_pages =
        ((addr + size - 1) >> Memory::PAGE_BITS) - (addr >> Memory::PAGE_BITS) + 1;
    const u32 page_start = addr >> Memory::PAGE_BITS;
    const u32 page_end = page_start + num_pages;

    // Interval maps will erase segments if count reaches 0, so if delta is negative we have to
    // subtract after iterating
    const auto pages_interval = PageMap::interval_type::right_open(page_start, page_end);
    if (delta > 0)
        cached_pages.add({pages_interval, delta});

    for (const auto& pair : RangeFromInterval(cached_pages, pages_interval)) {
        const auto interval = pair.first & pages_interval;
        const int count = pair.second;

        const PAddr interval_start_addr = boost::icl::first(interval) << Memory::PAGE_BITS;
        const PAddr interval_end_addr = boost::icl::last_next(interval) << Memory::PAGE_BITS;
        const u32 interval_size = interval_end_addr - interval_start_addr;

        if (delta > 0 && count == delta)
            VideoCore::g_memory->RasterizerMarkRegionCached(interval_start_addr, interval_size,
                                                            true);
        else if (delta < 0 && count == -delta)
            VideoCore::g_memory->RasterizerMarkRegionCached(interval_start_addr, interval_size,
                                                            false);
        else
            ASSERT(count >= 0);
    }

    if (delta < 0)
        cached_pages.add({pages_interval, delta});
}

} // namespace OpenGL
