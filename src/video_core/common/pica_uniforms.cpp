// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include "video_core/common/pica_uniforms.h"

namespace VideoCore {

void PicaUniformsData::SetFromRegs(const Pica::ShaderRegs& regs, const Pica::Shader::ShaderSetup& setup) {
    std::ranges::transform(setup.uniforms.b, bools.begin(), [](bool value) {
        return BoolAligned{value ? true : false};
    });

    std::ranges::transform(regs.int_uniforms, i.begin(), [](const auto& value) {
        return Common::Vec4u{value.x.Value(), value.y.Value(), value.z.Value(), value.w.Value()};
    });

    std::ranges::transform(setup.uniforms.f, f.begin(), [](const auto& value) {
        return Common::Vec4f{value.x.ToFloat32(), value.y.ToFloat32(),
                             value.z.ToFloat32(), value.w.ToFloat32()};
    });
}

} // namespace VideoCore
