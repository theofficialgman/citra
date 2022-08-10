// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "video_core/common/shader_gen.h"

namespace VideoCore::Vulkan {

class ShaderGenerator : public VideoCore::ShaderGeneratorBase {
public:
    ShaderGenerator() = default;
    ~ShaderGenerator() override = default;

    std::string GenerateTrivialVertexShader() override;
    std::string GenerateVertexShader(const Pica::Shader::ShaderSetup& setup, const PicaVSConfig& config) override;
    std::string GenerateFixedGeometryShader(const PicaFixedGSConfig& config) override;
    std::string GenerateFragmentShader(const PicaFSConfig& config) override;
};

} // namespace VideoCore
