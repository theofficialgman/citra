// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <optional>
#include <memory>
#include <unordered_map>
#include "video_core/regs.h"
#include "video_core/common/shader.h"

namespace Core {
class System;
}

namespace FileUtil {
class IOFile;
}

namespace VideoCore {

enum class ProgramType : u32 {
    VertexShader = 0,
    GeometryShader = 1,
    FragmentShader = 2
};

// Describes a shader how it's used by the Pica GPU
class ShaderDiskCacheRaw {
public:
    ShaderDiskCacheRaw() = default;
    ShaderDiskCacheRaw(u64 unique_identifier, ProgramType program_type, Pica::Regs config,
                       std::vector<u32> program_code) : unique_identifier(unique_identifier),
        program_type(program_type), config(config), program_code(program_code) {}
    ~ShaderDiskCacheRaw() = default;

    bool Load(FileUtil::IOFile& file);
    bool Save(FileUtil::IOFile& file) const;

    // Returns the unique hash of the program code and pica registers
    u64 GetUniqueIdentifier() const {
        return unique_identifier;
    }

    // Returns the shader program type
    ProgramType GetProgramType() const {
        return program_type;
    }

    // Returns an immutable span to the program code
    std::span<const u32> GetProgramCode() const {
        return program_code;
    }

    // Returns the pica register state used to generate the program code
    const Pica::Regs& GetRawShaderConfig() const {
        return config;
    }

private:
    u64 unique_identifier = 0;
    ProgramType program_type{};
    Pica::Regs config{};
    std::vector<u32> program_code{};
};

// Contains decompiled data from a shader
struct ShaderDiskCacheDecompiled {
    std::string result;
    bool sanitize_mul;
};

using ShaderDecompiledMap = std::unordered_map<u64, ShaderDiskCacheDecompiled>;

class BackendBase;

class ShaderDiskCache {
public:
    ShaderDiskCache(std::unique_ptr<BackendBase>& backend);
    ~ShaderDiskCache() = default;

    /// Loads transferable cache. If file has a old version or on failure, it deletes the file.
    std::optional<std::vector<ShaderDiskCacheRaw>> LoadTransferable();

    /// Removes the transferable (and precompiled) cache file.
    void InvalidateAll();

    /// Removes the precompiled cache file and clears virtual precompiled cache file.
    void InvalidatePrecompiled();

    /// Saves a raw dump to the transferable file. Checks for collisions.
    void SaveRaw(const ShaderDiskCacheRaw& entry);

    /// Saves a decompiled entry to the precompiled file. Does not check for collisions.
    void SaveDecompiled(u64 unique_identifier, const std::string& code, bool sanitize_mul);

    /// Loads the transferable cache. Returns empty on failure.
    std::optional<ShaderDecompiledMap> LoadPrecompiled();

private:
    /// Loads a decompiled cache entry from m_precompiled_cache_virtual_file.
    /// Returns empty on failure.
    std::optional<ShaderDiskCacheDecompiled> LoadDecompiledEntry();

    /// Saves a decompiled entry to the passed file. Does not check for collisions.
    void SaveDecompiledToFile(FileUtil::IOFile& file, u64 unique_identifier,
                              const std::string& code, bool sanitize_mul);

    /// Saves a decompiled entry to the virtual precompiled cache. Does not check for collisions.
    bool SaveDecompiledToCache(u64 unique_identifier, const std::string& code, bool sanitize_mul);

    /// Returns if the cache can be used
    bool IsUsable() const;

    /// Opens current game's transferable file and write it's header if it doesn't exist
    FileUtil::IOFile AppendTransferableFile();

    /// Save precompiled header to precompiled_cache_in_memory
    void SavePrecompiledHeaderToVirtualPrecompiledCache();

    /// Create shader disk cache directories. Returns true on success.
    bool EnsureDirectories() const;

    /// Gets current game's transferable file path
    std::string GetTransferablePath();

    /// Gets current game's precompiled file path
    std::string GetPrecompiledPath();

    /// Get user's transferable directory path
    std::string GetTransferableDir() const;

    /// Get user's precompiled directory path
    std::string GetPrecompiledDir() const;

    std::string GetPrecompiledShaderDir() const;

    /// Get user's shader directory path
    std::string GetBaseDir() const;

    /// Get current game's title id as u64
    u64 GetProgramID();

    /// Get current game's title id
    std::string GetTitleID();

    template <typename T>
    bool SaveArrayToPrecompiled(const T* data, std::size_t length) {
        const u8* data_view = reinterpret_cast<const u8*>(data);
        decompressed_precompiled_cache.insert(decompressed_precompiled_cache.end(), &data_view[0],
                                              &data_view[length * sizeof(T)]);
        decompressed_precompiled_cache_offset += length * sizeof(T);
        return true;
    }

    template <typename T>
    bool LoadArrayFromPrecompiled(T* data, std::size_t length) {
        u8* data_view = reinterpret_cast<u8*>(data);
        std::copy_n(decompressed_precompiled_cache.data() + decompressed_precompiled_cache_offset,
                    length * sizeof(T), data_view);
        decompressed_precompiled_cache_offset += length * sizeof(T);
        return true;
    }

    template <typename T>
    bool SaveObjectToPrecompiled(const T& object) {
        return SaveArrayToPrecompiled(&object, 1);
    }

    bool SaveObjectToPrecompiled(bool object) {
        const auto value = static_cast<u8>(object);
        return SaveArrayToPrecompiled(&value, 1);
    }

    template <typename T>
    bool LoadObjectFromPrecompiled(T& object) {
        return LoadArrayFromPrecompiled(&object, 1);
    }

private:
    std::unique_ptr<BackendBase>& backend;

    // Stores whole precompiled cache which will be read from or saved to the precompiled cache file
    std::vector<u8> decompressed_precompiled_cache;

    // Stores the current offset of the precompiled cache file for IO purposes
    std::size_t decompressed_precompiled_cache_offset = 0;

    // Stored transferable shaders
    std::unordered_map<u64, ShaderDiskCacheRaw> transferable;

    // The cache has been loaded at boot
    bool tried_to_load{};

    u64 program_id{};
    std::string title_id;

    FileUtil::IOFile AppendPrecompiledFile();
};

} // namespace OpenGL
