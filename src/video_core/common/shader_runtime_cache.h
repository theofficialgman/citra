// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include "video_core/common/backend.h"
#include "video_core/common/shader_gen.h"

namespace VideoCore {

using ShaderCacheResult = std::tuple<ShaderHandle, std::optional<std::string>>;

template <typename KeyType>
using ShaderGenerator = std::string(ShaderGeneratorBase::*)(const KeyType&);

template <typename KeyType, ShaderGenerator<KeyType> CodeGenerator, ShaderStage stage>
class ShaderCache {
public:
    ShaderCache(std::unique_ptr<BackendBase>& backend,
                std::unique_ptr<ShaderGeneratorBase>& generator) :
        backend(backend), generator(generator) {}
    ~ShaderCache() = default;

    // Returns a shader handle generated from the provided config
    ShaderCacheResult Get(const KeyType& config) {
        auto [iter, new_shader] = shaders.emplace(config, ShaderHandle{});
        ShaderHandle& shader = iter->second;

        if (new_shader) {
            auto result = (generator.get()->*CodeGenerator)(config);
            shader = backend->CreateShader(stage, "Cached shader", result);
            shader->Compile(ShaderOptimization::Debug); // TODO: Change this

            return std::make_tuple(shader, result);
        }

        return std::make_tuple(shader, std::nullopt);
    }

    void Inject(const KeyType& key, ShaderHandle&& shader) {
        shaders.emplace(key, std::move(shader));
    }

private:
    std::unique_ptr<BackendBase>& backend;
    std::unique_ptr<ShaderGeneratorBase>& generator;
    std::unordered_map<KeyType, ShaderHandle> shaders;
};

template <typename KeyType>
using PicaShaderGenerator = std::string (ShaderGeneratorBase::*)(const Pica::Shader::ShaderSetup&,
                                                                 const KeyType&);

/**
 * This is a cache designed for shaders translated from PICA shaders. The first cache matches the
 * config structure like a normal cache does. On cache miss, the second cache matches the generated
 * GLSL code. The configuration is like this because there might be leftover code in the PICA shader
 * program buffer from the previous shader, which is hashed into the config, resulting several
 * different config values from the same shader program.
 */
template <typename KeyType, PicaShaderGenerator<KeyType> CodeGenerator, ShaderStage stage>
class ShaderDoubleCache {
public:
    ShaderDoubleCache(std::unique_ptr<BackendBase>& backend,
                      std::unique_ptr<ShaderGeneratorBase>& generator) :
        backend(backend), generator(generator) {}
    ~ShaderDoubleCache() = default;

    ShaderCacheResult Get(const KeyType& key, const Pica::Shader::ShaderSetup& setup) {
        if (auto map_iter = shader_map.find(key); map_iter == shader_map.end()) {
            std::string program = (generator.get()->*CodeGenerator)(setup, key);
            auto [iter, new_shader] = shader_cache.emplace(program, ShaderHandle{});
            ShaderHandle& shader = iter->second;

            if (new_shader) {
                shader = backend->CreateShader(stage, "Cached shader", program);
                shader->Compile(ShaderOptimization::Debug); // TODO: Change this
            }

            shader_map[key] = &shader;
            return std::make_tuple(shader, std::move(program));
        } else {
            return std::make_tuple(*map_iter->second, std::nullopt);
        }
    }

    void Inject(const KeyType& key, std::string decomp, ShaderHandle&& program) {
        const auto iter = shader_cache.emplace(std::move(decomp), std::move(stage)).first;

        ShaderHandle& cached_shader = iter->second;
        shader_map.insert_or_assign(key, &cached_shader);
    }

private:
    std::unique_ptr<BackendBase>& backend;
    std::unique_ptr<ShaderGeneratorBase>& generator;
    std::unordered_map<KeyType, ShaderHandle*> shader_map;
    std::unordered_map<std::string, ShaderHandle> shader_cache;
};

// Define shader cache types for convenience
using FragmentShaders = ShaderCache<PicaFSConfig,
                                    &ShaderGeneratorBase::GenerateFragmentShader,
                                    ShaderStage::Fragment>;

using PicaVertexShaders = ShaderDoubleCache<PicaVSConfig,
                                            &ShaderGeneratorBase::GenerateVertexShader,
                                            ShaderStage::Vertex>;

using FixedGeometryShaders = ShaderCache<PicaFixedGSConfig,
                                         &ShaderGeneratorBase::GenerateFixedGeometryShader,
                                         ShaderStage::Geometry>;

} // namespace VideoCore
