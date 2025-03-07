// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cstring>
#include <fmt/format.h>
#include "common/common_paths.h"
#include "common/file_util.h"
#include "common/logging/log.h"
#include "common/scm_rev.h"
#include "common/zstd_compression.h"
#include "core/core.h"
#include "core/settings.h"
#include "video_core/common/backend.h"
#include "video_core/common/shader_disk_cache.h"

namespace VideoCore {

using ShaderCacheVersionHash = std::array<u8, 64>;

enum class TransferableEntryKind : u32 {
    Raw = 0,
};

enum class PrecompiledEntryKind : u32 {
    Decompiled = 0,
    Dump = 1,
};

constexpr u32 NativeVersion = 1;

ShaderCacheVersionHash GetShaderCacheVersionHash() {
    ShaderCacheVersionHash hash{};
    const std::size_t length = std::min(std::strlen(Common::g_shader_cache_version), hash.size());
    std::memcpy(hash.data(), Common::g_shader_cache_version, length);
    return hash;
}

bool ShaderDiskCacheRaw::Load(FileUtil::IOFile& file) {
    if (file.ReadBytes(&unique_identifier, sizeof(u64)) != sizeof(u64) ||
        file.ReadBytes(&program_type, sizeof(u32)) != sizeof(u32)) {
        return false;
    }

    u64 reg_array_len{};
    if (file.ReadBytes(&reg_array_len, sizeof(u64)) != sizeof(u64)) {
        return false;
    }

    if (file.ReadArray(config.reg_array.data(), reg_array_len) != reg_array_len) {
        return false;
    }

    // Read in type specific configuration
    if (program_type == ProgramType::VertexShader) {
        u64 code_len{};
        if (file.ReadBytes(&code_len, sizeof(u64)) != sizeof(u64)) {
            return false;
        }

        program_code.resize(code_len);
        if (file.ReadArray(program_code.data(), code_len) != code_len) {
            return false;
        }
    }

    return true;
}

bool ShaderDiskCacheRaw::Save(FileUtil::IOFile& file) const {
    if (file.WriteObject(unique_identifier) != 1 ||
        file.WriteObject(static_cast<u32>(program_type)) != 1) {
        return false;
    }

    // Just for future proofing, save the sizes of the array to the file
    const std::size_t reg_array_len = Pica::Regs::NUM_REGS;
    if (file.WriteObject(static_cast<u64>(reg_array_len)) != 1) {
        return false;
    }
    if (file.WriteArray(config.reg_array.data(), reg_array_len) != reg_array_len) {
        return false;
    }

    if (program_type == ProgramType::VertexShader) {
        const std::size_t code_len = program_code.size();
        if (file.WriteObject(static_cast<u64>(code_len)) != 1) {
            return false;
        }
        if (file.WriteArray(program_code.data(), code_len) != code_len) {
            return false;
        }
    }
    return true;
}

ShaderDiskCache::ShaderDiskCache(std::unique_ptr<BackendBase>& backend) : backend(backend) {

}

std::optional<std::vector<ShaderDiskCacheRaw>> ShaderDiskCache::LoadTransferable() {
    const bool has_title_id = GetProgramID() != 0;
    if (!Settings::values.use_hw_shader || !Settings::values.use_disk_shader_cache ||
        !has_title_id) {
        return std::nullopt;
    }
    tried_to_load = true;

    FileUtil::IOFile file{GetTransferablePath(), "rb"};
    if (!file.IsOpen()) {
        LOG_INFO(Render_Vulkan, "No transferable shader cache found for game with title id={}",
                 GetTitleID());
        return std::nullopt;
    }

    u32 version{};
    if (file.ReadBytes(&version, sizeof(version)) != sizeof(version)) {
        LOG_ERROR(Render_Vulkan, "Failed to get transferable cache version for title id={} - skipping",
                  GetTitleID());
        return std::nullopt;
    }

    if (version < NativeVersion) {
        LOG_INFO(Render_Vulkan, "Transferable shader cache is old - removing");
        file.Close();
        InvalidateAll();

        return std::nullopt;
    }
    if (version > NativeVersion) {
        LOG_WARNING(Render_Vulkan, "Transferable shader cache was generated with a newer version "
                                   "of the emulator - skipping");
        return std::nullopt;
    }

    // Version is valid, load the shaders
    std::vector<ShaderDiskCacheRaw> raws;
    while (file.Tell() < file.GetSize()) {
        TransferableEntryKind kind{};
        if (file.ReadBytes(&kind, sizeof(u32)) != sizeof(u32)) {
            LOG_ERROR(Render_Vulkan, "Failed to read transferable file - skipping");
            return std::nullopt;
        }

        switch (kind) {
        case TransferableEntryKind::Raw: {
            ShaderDiskCacheRaw entry;
            if (!entry.Load(file)) {
                LOG_ERROR(Render_Vulkan, "Failed to load transferable raw entry - skipping");
                return std::nullopt;
            }
            transferable.emplace(entry.GetUniqueIdentifier(), ShaderDiskCacheRaw{});
            raws.push_back(std::move(entry));
            break;
        }
        default:
            LOG_ERROR(Render_OpenGL, "Unknown transferable shader cache entry kind={} - skipping",
                      kind);
            return std::nullopt;
        }
    }

    LOG_INFO(Render_OpenGL, "Found a transferable disk cache with {} entries", raws.size());
    return {std::move(raws)};
}

std::optional<ShaderDecompiledMap> ShaderDiskCache::LoadPrecompiled() {
    if (!IsUsable()) {
        return std::nullopt;
    }

    FileUtil::IOFile file(GetPrecompiledPath(), "rb");
    if (!file.IsOpen()) {
        LOG_INFO(Render_OpenGL, "No precompiled shader cache found for game with title id={}",
                 GetTitleID());
        return {};
    }

    auto Invalidate = [&file, this]() -> std::nullopt_t {
        LOG_INFO(Render_OpenGL, "Failed to load precompiled cache for game with title id={} - removing",
                 GetTitleID());
        file.Close();
        InvalidatePrecompiled();

        return std::nullopt;
    };

    // Read compressed file from disk and decompress to virtual precompiled cache file
    std::vector<u8> precompiled_file(file.GetSize());
    file.ReadBytes(precompiled_file.data(), precompiled_file.size());

    const auto decompressed = Common::Compression::DecompressDataZSTD(precompiled_file);
    SaveArrayToPrecompiled(decompressed.data(), decompressed.size());
    decompressed_precompiled_cache_offset = 0;

    ShaderCacheVersionHash file_hash{};
    if (!LoadArrayFromPrecompiled(file_hash.data(), file_hash.size())) {
        return std::nullopt;
    }

    if (GetShaderCacheVersionHash() != file_hash) {
        LOG_INFO(Render_OpenGL, "Precompiled cache is from another version of the emulator");
        return std::nullopt;
    }

    ShaderDecompiledMap decompiled{};
    while (decompressed_precompiled_cache_offset < decompressed_precompiled_cache.size()) {
        PrecompiledEntryKind kind{};
        if (!LoadObjectFromPrecompiled(kind)) {
            return Invalidate();
        }

        switch (kind) {
        case PrecompiledEntryKind::Decompiled: {
            u64 unique_identifier{};
            if (!LoadObjectFromPrecompiled(unique_identifier)) {
                return Invalidate();
            }

            std::optional entry = LoadDecompiledEntry();
            if (!entry) {
                return Invalidate();
            }

            decompiled.insert({unique_identifier, std::move(*entry)});
            break;
        }
        default:
            return Invalidate();
        }
    }

    LOG_INFO(Render_OpenGL, "Found a precompiled disk cache with {} decompiled entries", decompiled.size());
    return decompiled;
}

std::optional<ShaderDiskCacheDecompiled> ShaderDiskCache::LoadDecompiledEntry() {

    bool sanitize_mul;
    if (!LoadObjectFromPrecompiled(sanitize_mul)) {
        return std::nullopt;
    }

    u32 code_size{};
    if (!LoadObjectFromPrecompiled(code_size)) {
        return std::nullopt;
    }

    std::string code(code_size, '\0');
    if (!LoadArrayFromPrecompiled(code.data(), code.size())) {
        return std::nullopt;
    }

    const ShaderDiskCacheDecompiled entry = {
        .result = std::move(code),
        .sanitize_mul = sanitize_mul
    };

    return entry;
}

void ShaderDiskCache::SaveDecompiledToFile(FileUtil::IOFile& file, u64 unique_identifier,
                                           const std::string& result,
                                           bool sanitize_mul) {
    if (!IsUsable())
        return;

    if (file.WriteObject(static_cast<u32>(PrecompiledEntryKind::Decompiled)) != 1 ||
        file.WriteObject(unique_identifier) != 1 || file.WriteObject(sanitize_mul) != 1 ||
        file.WriteObject(static_cast<u32>(result.size())) != 1 ||
        file.WriteArray(result.data(), result.size()) != result.size()) {
        LOG_ERROR(Render_OpenGL, "Failed to save decompiled cache entry - removing");

        file.Close();
        InvalidatePrecompiled();
    }
}

bool ShaderDiskCache::SaveDecompiledToCache(u64 unique_identifier, const std::string& result,
                                            bool sanitize_mul) {
    if (!SaveObjectToPrecompiled(static_cast<u32>(PrecompiledEntryKind::Decompiled)) ||
        !SaveObjectToPrecompiled(unique_identifier) || !SaveObjectToPrecompiled(sanitize_mul) ||
        !SaveObjectToPrecompiled(static_cast<u32>(result.size())) ||
        !SaveArrayToPrecompiled(result.data(), result.size())) {
        return false;
    }

    return true;
}

void ShaderDiskCache::InvalidateAll() {
    if (!FileUtil::Delete(GetTransferablePath())) {
        LOG_ERROR(Render_OpenGL, "Failed to invalidate transferable file={}",
                  GetTransferablePath());
    }
    InvalidatePrecompiled();
}

void ShaderDiskCache::InvalidatePrecompiled() {
    // Clear virtual precompiled cache file
    decompressed_precompiled_cache.resize(0);

    if (!FileUtil::Delete(GetPrecompiledPath())) {
        LOG_ERROR(Render_OpenGL, "Failed to invalidate precompiled file={}", GetPrecompiledPath());
    }
}

void ShaderDiskCache::SaveRaw(const ShaderDiskCacheRaw& entry) {
    if (!IsUsable())
        return;

    const u64 id = entry.GetUniqueIdentifier();
    if (transferable.find(id) != transferable.end()) {
        // The shader already exists
        return;
    }

    FileUtil::IOFile file = AppendTransferableFile();
    if (!file.IsOpen()) {
        return;
    }

    if (file.WriteObject(TransferableEntryKind::Raw) != 1 || !entry.Save(file)) {
        LOG_ERROR(Render_OpenGL, "Failed to save raw transferable cache entry - removing");
        file.Close();
        InvalidateAll();
        return;
    }

    transferable.insert({id, entry});
}

void ShaderDiskCache::SaveDecompiled(u64 unique_identifier, const std::string& code, bool sanitize_mul) {
    if (!IsUsable())
        return;

    if (decompressed_precompiled_cache.empty()) {
        SavePrecompiledHeaderToVirtualPrecompiledCache();
    }

    if (!SaveDecompiledToCache(unique_identifier, code, sanitize_mul)) {
        LOG_ERROR(Render_OpenGL,
                  "Failed to save decompiled entry to the precompiled file - removing");
        InvalidatePrecompiled();
    }
}

bool ShaderDiskCache::IsUsable() const {
    return tried_to_load && Settings::values.use_disk_shader_cache;
}

FileUtil::IOFile ShaderDiskCache::AppendTransferableFile() {
    if (!EnsureDirectories())
        return {};

    const auto transferable_path{GetTransferablePath()};
    const bool existed = FileUtil::Exists(transferable_path);

    FileUtil::IOFile file(transferable_path, "ab");
    if (!file.IsOpen()) {
        LOG_ERROR(Render_OpenGL, "Failed to open transferable cache in path={}", transferable_path);
        return {};
    }
    if (!existed || file.GetSize() == 0) {
        // If the file didn't exist, write its version
        if (file.WriteObject(NativeVersion) != 1) {
            LOG_ERROR(Render_OpenGL, "Failed to write transferable cache version in path={}",
                      transferable_path);
            return {};
        }
    }
    return file;
}

FileUtil::IOFile ShaderDiskCache::AppendPrecompiledFile() {
    if (!EnsureDirectories())
        return {};

    const auto precompiled_path{GetPrecompiledPath()};
    const bool existed = FileUtil::Exists(precompiled_path);

    FileUtil::IOFile file(precompiled_path, "ab");
    if (!file.IsOpen()) {
        LOG_ERROR(Render_OpenGL, "Failed to open precompiled cache in path={}", precompiled_path);
        return {};
    }
    if (!existed || file.GetSize() == 0) {
        // If the file didn't exist, write its version
        const auto hash{GetShaderCacheVersionHash()};
        if (file.WriteArray(hash.data(), hash.size()) != hash.size()) {
            LOG_ERROR(Render_OpenGL, "Failed to write precompiled cache version in path={}",
                      precompiled_path);
            return {};
        }
    }
    return file;
}

void ShaderDiskCache::SavePrecompiledHeaderToVirtualPrecompiledCache() {
    const ShaderCacheVersionHash hash = GetShaderCacheVersionHash();
    if (!SaveArrayToPrecompiled(hash.data(), hash.size())) {
        LOG_ERROR(Render_OpenGL,
                  "Failed to write precompiled cache version hash to virtual precompiled cache file");
    }
}

bool ShaderDiskCache::EnsureDirectories() const {
    const auto CreateDir = [](const std::string& dir) {
        if (!FileUtil::CreateDir(dir)) {
            LOG_ERROR(Render_OpenGL, "Failed to create directory={}", dir);
            return false;
        }
        return true;
    };

    return CreateDir(FileUtil::GetUserPath(FileUtil::UserPath::ShaderDir)) &&
           CreateDir(GetBaseDir()) && CreateDir(GetTransferableDir()) &&
           CreateDir(GetPrecompiledDir()) && CreateDir(GetPrecompiledShaderDir());
}

std::string ShaderDiskCache::GetTransferablePath() {
    return FileUtil::SanitizePath(GetTransferableDir() + DIR_SEP_CHR + GetTitleID() + ".bin");
}

std::string ShaderDiskCache::GetPrecompiledPath() {
    return FileUtil::SanitizePath(GetPrecompiledShaderDir() + DIR_SEP_CHR + GetTitleID() + ".bin");
}

std::string ShaderDiskCache::GetTransferableDir() const {
    return GetBaseDir() + DIR_SEP "transferable";
}

std::string ShaderDiskCache::GetPrecompiledDir() const {
    return GetBaseDir() + DIR_SEP "precompiled";
}

std::string ShaderDiskCache::GetPrecompiledShaderDir() const {
    return GetPrecompiledDir() + DIR_SEP "separable";
}

std::string ShaderDiskCache::GetBaseDir() const {
    return FileUtil::GetUserPath(FileUtil::UserPath::ShaderDir) + DIR_SEP "opengl";
}

u64 ShaderDiskCache::GetProgramID() {
    // Skip games without title id
    if (program_id != 0) {
        return program_id;
    }
    if (Core::System::GetInstance().GetAppLoader().ReadProgramId(program_id) !=
        Loader::ResultStatus::Success) {
        return 0;
    }
    return program_id;
}

std::string ShaderDiskCache::GetTitleID() {
    if (!title_id.empty()) {
        return title_id;
    }
    title_id = fmt::format("{:016X}", GetProgramID());
    return title_id;
}

} // namespace VideoCore
