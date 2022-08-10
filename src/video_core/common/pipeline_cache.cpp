// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <thread>
#include <mutex>
#include "core/frontend/scope_acquire_context.h"
#include "video_core/common/pipeline_cache.h"
#include "video_core/common/shader.h"
#include "video_core/common/shader_gen.h"
#include "video_core/renderer_vulkan/vk_shader_gen.h"
#include "video_core/video_core.h"

namespace VideoCore {

static u64 GetUniqueIdentifier(const Pica::Regs& regs, std::span<const u32> code) {
    u64 hash = Common::ComputeHash64(regs.reg_array.data(), Pica::Regs::NUM_REGS * sizeof(u32));

    if (code.size() > 0) {
        u64 code_uid = Common::ComputeHash64(code.data(), code.size() * sizeof(u32));
        hash = Common::HashCombine(hash, code_uid);
    }

    return hash;
}

static auto BuildVSConfigFromRaw(const ShaderDiskCacheRaw& raw) {
    Pica::Shader::ProgramCode program_code{};
    Pica::Shader::SwizzleData swizzle_data{};
    std::copy_n(raw.GetProgramCode().begin(), Pica::Shader::MAX_PROGRAM_CODE_LENGTH,
                program_code.begin());
    std::copy_n(raw.GetProgramCode().begin() + Pica::Shader::MAX_PROGRAM_CODE_LENGTH,
                Pica::Shader::MAX_SWIZZLE_DATA_LENGTH, swizzle_data.begin());
    Pica::Shader::ShaderSetup setup;
    setup.program_code = program_code;
    setup.swizzle_data = swizzle_data;
    return std::make_tuple(PicaVSConfig{raw.GetRawShaderConfig().vs, setup}, setup);
}

PipelineCache::PipelineCache(Frontend::EmuWindow& emu_window, std::unique_ptr<BackendBase>& backend)
    : emu_window(emu_window), backend(backend), pica_vertex_shaders(backend, generator),
      fixed_geometry_shaders(backend, generator), fragment_shaders(backend, generator),
      disk_cache(backend) {
    // TODO: Don't hardcode this!
    generator = std::make_unique<Vulkan::ShaderGenerator>();
}

PipelineHandle PipelineCache::GetPipeline(PipelineInfo& info) {
    // Update shader handles
    info.shaders[static_cast<u32>(ProgramType::VertexShader)] = current_vertex_shader;
    info.shaders[static_cast<u32>(ProgramType::GeometryShader)] = current_geometry_shader;
    info.shaders[static_cast<u32>(ProgramType::FragmentShader)] = current_fragment_shader;

    // Search cache
    const u64 pipeline_hash = backend->PipelineInfoHash(info);
    if (auto iter = cached_pipelines.find(pipeline_hash); iter != cached_pipelines.end()) {
        return iter->second;
    }

    // Create new pipeline
    auto iter = cached_pipelines.emplace(info, backend->CreatePipeline(PipelineType::Graphics, info)).first;
    return iter->second;
}

bool PipelineCache::UsePicaVertexShader(const Pica::Regs& regs, Pica::Shader::ShaderSetup& setup) {
    PicaVSConfig config{regs.vs, setup};
    auto [handle, shader_str] = pica_vertex_shaders.Get(config, setup);
    if (!handle.IsValid()) {
        return false;
    }

    current_vertex_shader = handle;

    // Save VS to the disk cache if its a new shader
    if (shader_str.has_value()) {
        // Copy program code
        std::vector<u32> program_code{setup.program_code.begin(), setup.program_code.end()};
        program_code.insert(program_code.end(), setup.swizzle_data.begin(), setup.swizzle_data.end());

        // Hash the bytecode and save the pica program
        const u64 unique_identifier = GetUniqueIdentifier(regs, program_code);
        const ShaderDiskCacheRaw raw{unique_identifier, ProgramType::VertexShader,
                                     regs, std::move(program_code)};

        disk_cache.SaveRaw(raw);
        disk_cache.SaveDecompiled(unique_identifier, shader_str.value(),
                                  VideoCore::g_hw_shader_accurate_mul);
    }

    return true;
}

void PipelineCache::UseTrivialVertexShader() {
    current_vertex_shader = trivial_vertex_shader;
}

void PipelineCache::UseFixedGeometryShader(const Pica::Regs& regs) {
    PicaFixedGSConfig gs_config{regs};
    auto [handle, _] = fixed_geometry_shaders.Get(gs_config);
    current_geometry_shader = handle;
}

void PipelineCache::UseTrivialGeometryShader() {
    current_geometry_shader = ShaderHandle{};
}

void PipelineCache::UseFragmentShader(const Pica::Regs& regs) {
    PicaFSConfig config{regs};
    auto [handle, shader_str] = fragment_shaders.Get(config);
    current_fragment_shader = handle;

    // Save FS to the disk cache if its a new shader
    if (shader_str.has_value()) {
        u64 unique_identifier = GetUniqueIdentifier(regs, {});
        ShaderDiskCacheRaw raw{unique_identifier, ProgramType::FragmentShader, regs, {}};
        disk_cache.SaveRaw(raw);
        disk_cache.SaveDecompiled(unique_identifier, shader_str.value(), false);
    }
}

void PipelineCache::LoadDiskCache(const std::atomic_bool& stop_loading, const DiskLoadCallback& callback) {
    const auto transferable = disk_cache.LoadTransferable();
    if (!transferable.has_value()) {
        return;
    }

    const auto& raws = transferable.value();

    // Load uncompressed precompiled file for non-separable shaders.
    // Precompiled file for separable shaders is compressed.
    std::optional decompiled = disk_cache.LoadPrecompiled();
    if (stop_loading) {
        return;
    }

    std::mutex mutex;
    std::atomic_bool compilation_failed = false;
    if (callback) {
        callback(VideoCore::LoadCallbackStage::Decompile, 0, raws.size());
    }

    std::vector<std::size_t> load_raws_index;
    for (u64 i = 0; i < raws.size(); i++) {
        if (stop_loading || compilation_failed) {
            break;
        }

        const ShaderDiskCacheRaw& raw = raws[i];
        const u64 unique_identifier = raw.GetUniqueIdentifier();
        const u64 calculated_hash = GetUniqueIdentifier(raw.GetRawShaderConfig(), raw.GetProgramCode());

        // Check for any data corruption
        if (unique_identifier != calculated_hash) {
            LOG_ERROR(Render_Vulkan, "Invalid hash in entry={:016x} (obtained hash={:016x}) - removing "
                                     "shader cache",
                                     raw.GetUniqueIdentifier(), calculated_hash);

            disk_cache.InvalidateAll();
            break;
        }

        const auto iter = decompiled->find(unique_identifier);

        ShaderHandle shader{};
        if (iter != decompiled->end()) {
            // Only load the vertex shader if its sanitize_mul setting matches
            ShaderDiskCacheDecompiled& decomp = iter->second;
            if (raw.GetProgramType() == ProgramType::VertexShader &&
                decomp.sanitize_mul != VideoCore::g_hw_shader_accurate_mul) {
                break;
            }

            ShaderStage stage;
            switch (raw.GetProgramType()) {
            case ProgramType::VertexShader:
                stage = ShaderStage::Vertex;
                break;
            case ProgramType::GeometryShader:
                stage = ShaderStage::Geometry;
                break;
            case ProgramType::FragmentShader:
                stage = ShaderStage::Fragment;
                break;
            }

            // Create shader from GLSL source
            shader = backend->CreateShader(stage, "Precompiled shader", decomp.result);

            // We have both the binary shader and the decompiled, so inject it into the
            // cache
            if (raw.GetProgramType() == ProgramType::VertexShader) {
                auto [conf, setup] = BuildVSConfigFromRaw(raw);
                std::scoped_lock lock(mutex);
                pica_vertex_shaders.Inject(conf, decomp.result, std::move(shader));

            } else if (raw.GetProgramType() == ProgramType::FragmentShader) {
                const PicaFSConfig conf{raw.GetRawShaderConfig()};
                std::scoped_lock lock(mutex);
                fragment_shaders.Inject(conf, std::move(shader));

            } else {
                // Unsupported shader type got stored somehow so nuke the cache
                LOG_CRITICAL(Frontend, "failed to load raw ProgramType {}", raw.GetProgramType());
                compilation_failed = true;
                break;
            }
        } else {
            // Since precompiled didn't have the dump, we'll load them in the next phase
            std::scoped_lock lock(mutex);
            load_raws_index.push_back(i);
        }

        if (callback) {
            callback(VideoCore::LoadCallbackStage::Decompile, i, raws.size());
        }
    }


    // Invalidate the precompiled cache if a shader dumped shader was rejected
    bool load_all_raws = false;
    if (compilation_failed) {
        disk_cache.InvalidatePrecompiled();
        load_all_raws = true;
    }

    const std::size_t load_raws_size = load_all_raws ? raws.size() : load_raws_index.size();

    if (callback) {
        callback(VideoCore::LoadCallbackStage::Build, 0, load_raws_size);
    }

    compilation_failed = false;

    std::size_t built_shaders = 0; // It doesn't have be atomic since it's used behind a mutex
    const auto LoadRawSepareble = [&](Frontend::GraphicsContext* context, std::size_t begin,
                                      std::size_t end) {
        Frontend::ScopeAcquireContext scope(*context);
        for (u64 i = begin; i < end; ++i) {
            if (stop_loading || compilation_failed) {
                return;
            }

            const u64 raws_index = load_all_raws ? i : load_raws_index[i];
            const auto& raw = raws[raws_index];
            const u64 unique_identifier = raw.GetUniqueIdentifier();

            bool sanitize_mul = false;
            ShaderHandle shader{nullptr};
            std::optional<std::string> result;

            // Otherwise decompile and build the shader at boot and save the result to the
            // precompiled file
            if (raw.GetProgramType() == ProgramType::VertexShader) {
                auto [conf, setup] = BuildVSConfigFromRaw(raw);
                result = generator->GenerateVertexShader(setup, conf);

                // Compile shader
                shader = backend->CreateShader(ShaderStage::Vertex, "Vertex shader", result.value());
                shader->Compile(ShaderOptimization::Debug);

                sanitize_mul = conf.sanitize_mul;
                std::scoped_lock lock(mutex);
                pica_vertex_shaders.Inject(conf, result.value(), std::move(shader));

            } else if (raw.GetProgramType() == ProgramType::FragmentShader) {
                const PicaFSConfig conf{raw.GetRawShaderConfig()};
                result = generator->GenerateFragmentShader(conf);

                // Compile shader
                shader = backend->CreateShader(ShaderStage::Fragment, "Fragment shader", result.value());
                shader->Compile(ShaderOptimization::Debug);

                std::scoped_lock lock(mutex);
                fragment_shaders.Inject(conf, std::move(shader));

            } else {
                // Unsupported shader type got stored somehow so nuke the cache
                LOG_ERROR(Frontend, "Failed to load raw ProgramType {}", raw.GetProgramType());
                compilation_failed = true;
                return;
            }

            if (!shader.IsValid()) {
                LOG_ERROR(Frontend, "Compilation from raw failed {:x} {:x}",
                          raw.GetProgramCode()[0], raw.GetProgramCode()[1]);
                compilation_failed = true;
                return;
            }

            std::scoped_lock lock(mutex);

            // If this is a new separable shader, add it the precompiled cache
            if (result) {
                disk_cache.SaveDecompiled(unique_identifier, *result, sanitize_mul);
            }

            if (callback) {
                callback(VideoCore::LoadCallbackStage::Build, ++built_shaders, load_raws_size);
            }
        }
    };

    const std::size_t num_workers{std::max(1U, std::thread::hardware_concurrency())};
    const std::size_t bucket_size{load_raws_size / num_workers};
    std::vector<std::unique_ptr<Frontend::GraphicsContext>> contexts(num_workers);
    std::vector<std::thread> threads(num_workers);

    for (std::size_t i = 0; i < num_workers; ++i) {
        const bool is_last_worker = i + 1 == num_workers;
        const std::size_t start{bucket_size * i};
        const std::size_t end{is_last_worker ? load_raws_size : start + bucket_size};

        // On some platforms the shared context has to be created from the GUI thread
        //contexts[i] = emu_window.CreateSharedContext();
        threads[i] = std::thread(LoadRawSepareble, contexts[i].get(), start, end);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    if (compilation_failed) {
        disk_cache.InvalidateAll();
    }
}

} // namespace VideoCore
