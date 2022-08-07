// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/bit_set.h"
#include "video_core/video_core.h"
#include "video_core/common/shader_gen.h"

namespace VideoCore {

PicaFSConfig::PicaFSConfig(const Pica::Regs& regs) {
    scissor_test_mode = regs.rasterizer.scissor_test.mode;
    depthmap_enable = regs.rasterizer.depthmap_enable;
    alpha_test_func = regs.framebuffer.output_merger.alpha_test.enable
                                ? regs.framebuffer.output_merger.alpha_test.func.Value()
                                : Pica::CompareFunc::Always;
    texture0_type = regs.texturing.texture0.type;
    texture2_use_coord1 = regs.texturing.main_config.texture2_use_coord1 != 0;

    // We don't need these otherwise, reset them to avoid unnecessary shader generation
    alphablend_enable = {};
    logic_op = {};

    // Copy relevant tev stages fields.
    // We don't sync const_color here because of the high variance, it is a
    // shader uniform instead.
    const auto stages = regs.texturing.GetTevStages();
    DEBUG_ASSERT(state.tev_stages.size() == tev_stages.size());
    for (std::size_t i = 0; i < stages.size(); i++) {
        const auto& tev_stage = stages[i];
        tev_stages[i].sources_raw = tev_stage.sources_raw;
        tev_stages[i].modifiers_raw = tev_stage.modifiers_raw;
        tev_stages[i].ops_raw = tev_stage.ops_raw;
        tev_stages[i].scales_raw = tev_stage.scales_raw;
    }

    fog_mode = regs.texturing.fog_mode;
    fog_flip = regs.texturing.fog_flip != 0;

    combiner_buffer_input = regs.texturing.tev_combiner_buffer_input.update_mask_rgb.Value() |
                                  regs.texturing.tev_combiner_buffer_input.update_mask_a.Value()
                                      << 4;

    // Fragment lighting
    lighting.enable = !regs.lighting.disable;
    lighting.src_num = regs.lighting.max_light_index + 1;

    for (u32 light_index = 0; light_index < lighting.src_num; ++light_index) {
        u32 num = regs.lighting.light_enable.GetNum(light_index);
        const auto& light = regs.lighting.light[num];
        auto& dst_light = lighting.light[light_index];

        dst_light.num = num;
        dst_light.directional = light.config.directional != 0;
        dst_light.two_sided_diffuse = light.config.two_sided_diffuse != 0;
        dst_light.geometric_factor_0 = light.config.geometric_factor_0 != 0;
        dst_light.geometric_factor_1 = light.config.geometric_factor_1 != 0;
        dst_light.dist_atten_enable = !regs.lighting.IsDistAttenDisabled(num);
        dst_light.spot_atten_enable = !regs.lighting.IsSpotAttenDisabled(num);
        dst_light.shadow_enable = !regs.lighting.IsShadowDisabled(num);
    }

    lighting.lut_d0.enable = regs.lighting.config1.disable_lut_d0 == 0;
    lighting.lut_d0.abs_input = regs.lighting.abs_lut_input.disable_d0 == 0;
    lighting.lut_d0.type = regs.lighting.lut_input.d0.Value();
    lighting.lut_d0.scale = regs.lighting.lut_scale.GetScale(regs.lighting.lut_scale.d0);

    lighting.lut_d1.enable = regs.lighting.config1.disable_lut_d1 == 0;
    lighting.lut_d1.abs_input = regs.lighting.abs_lut_input.disable_d1 == 0;
    lighting.lut_d1.type = regs.lighting.lut_input.d1.Value();
    lighting.lut_d1.scale = regs.lighting.lut_scale.GetScale(regs.lighting.lut_scale.d1);

    // This is a dummy field due to lack of the corresponding register
    lighting.lut_sp.enable = true;
    lighting.lut_sp.abs_input = regs.lighting.abs_lut_input.disable_sp == 0;
    lighting.lut_sp.type = regs.lighting.lut_input.sp.Value();
    lighting.lut_sp.scale = regs.lighting.lut_scale.GetScale(regs.lighting.lut_scale.sp);

    lighting.lut_fr.enable = regs.lighting.config1.disable_lut_fr == 0;
    lighting.lut_fr.abs_input = regs.lighting.abs_lut_input.disable_fr == 0;
    lighting.lut_fr.type = regs.lighting.lut_input.fr.Value();
    lighting.lut_fr.scale = regs.lighting.lut_scale.GetScale(regs.lighting.lut_scale.fr);

    lighting.lut_rr.enable = regs.lighting.config1.disable_lut_rr == 0;
    lighting.lut_rr.abs_input = regs.lighting.abs_lut_input.disable_rr == 0;
    lighting.lut_rr.type = regs.lighting.lut_input.rr.Value();
    lighting.lut_rr.scale = regs.lighting.lut_scale.GetScale(regs.lighting.lut_scale.rr);

    lighting.lut_rg.enable = regs.lighting.config1.disable_lut_rg == 0;
    lighting.lut_rg.abs_input = regs.lighting.abs_lut_input.disable_rg == 0;
    lighting.lut_rg.type = regs.lighting.lut_input.rg.Value();
    lighting.lut_rg.scale = regs.lighting.lut_scale.GetScale(regs.lighting.lut_scale.rg);

    lighting.lut_rb.enable = regs.lighting.config1.disable_lut_rb == 0;
    lighting.lut_rb.abs_input = regs.lighting.abs_lut_input.disable_rb == 0;
    lighting.lut_rb.type = regs.lighting.lut_input.rb.Value();
    lighting.lut_rb.scale = regs.lighting.lut_scale.GetScale(regs.lighting.lut_scale.rb);

    lighting.config = regs.lighting.config0.config;
    lighting.enable_primary_alpha = regs.lighting.config0.enable_primary_alpha;
    lighting.enable_secondary_alpha = regs.lighting.config0.enable_secondary_alpha;
    lighting.bump_mode = regs.lighting.config0.bump_mode;
    lighting.bump_selector = regs.lighting.config0.bump_selector;
    lighting.bump_renorm = regs.lighting.config0.disable_bump_renorm == 0;
    lighting.clamp_highlights = regs.lighting.config0.clamp_highlights != 0;

    lighting.enable_shadow = regs.lighting.config0.enable_shadow != 0;
    lighting.shadow_primary = regs.lighting.config0.shadow_primary != 0;
    lighting.shadow_secondary = regs.lighting.config0.shadow_secondary != 0;
    lighting.shadow_invert = regs.lighting.config0.shadow_invert != 0;
    lighting.shadow_alpha = regs.lighting.config0.shadow_alpha != 0;
    lighting.shadow_selector = regs.lighting.config0.shadow_selector;

    proctex.enable = regs.texturing.main_config.texture3_enable;
    if (proctex.enable) {
        proctex.coord = regs.texturing.main_config.texture3_coordinates;
        proctex.u_clamp = regs.texturing.proctex.u_clamp;
        proctex.v_clamp = regs.texturing.proctex.v_clamp;
        proctex.color_combiner = regs.texturing.proctex.color_combiner;
        proctex.alpha_combiner = regs.texturing.proctex.alpha_combiner;
        proctex.separate_alpha = regs.texturing.proctex.separate_alpha;
        proctex.noise_enable = regs.texturing.proctex.noise_enable;
        proctex.u_shift = regs.texturing.proctex.u_shift;
        proctex.v_shift = regs.texturing.proctex.v_shift;
        proctex.lut_width = regs.texturing.proctex_lut.width;
        proctex.lut_offset0 = regs.texturing.proctex_lut_offset.level0;
        proctex.lut_offset1 = regs.texturing.proctex_lut_offset.level1;
        proctex.lut_offset2 = regs.texturing.proctex_lut_offset.level2;
        proctex.lut_offset3 = regs.texturing.proctex_lut_offset.level3;
        proctex.lod_min = regs.texturing.proctex_lut.lod_min;
        proctex.lod_max = regs.texturing.proctex_lut.lod_max;
        proctex.lut_filter = regs.texturing.proctex_lut.filter;
    }

    shadow_rendering = regs.framebuffer.output_merger.fragment_operation_mode ==
                             Pica::FragmentOperationMode::Shadow;

    shadow_texture_orthographic = regs.texturing.shadow.orthographic != 0;
}

PicaVSConfig::PicaVSConfig(const Pica::ShaderRegs& regs, Pica::Shader::ShaderSetup& setup) {
    program_hash = setup.GetProgramCodeHash();
    swizzle_hash = setup.GetSwizzleDataHash();
    main_offset = regs.main_offset;
    sanitize_mul = VideoCore::g_hw_shader_accurate_mul;

    num_outputs = 0;
    output_map.fill(16);

    for (int reg : Common::BitSet<u32>(regs.output_mask)) {
        output_map[reg] = num_outputs++;
    }
}

PicaFixedGSConfig::PicaFixedGSConfig(const Pica::Regs& regs) {
    vs_output_attributes = Common::BitSet<u32>(regs.vs.output_mask).Count();
    gs_output_attributes = vs_output_attributes;

    semantic_maps.fill({16, 0});
    for (u32 attrib = 0; attrib < regs.rasterizer.vs_output_total; ++attrib) {
        const std::array semantics = {
            regs.rasterizer.vs_output_attributes[attrib].map_x.Value(),
            regs.rasterizer.vs_output_attributes[attrib].map_y.Value(),
            regs.rasterizer.vs_output_attributes[attrib].map_z.Value(),
            regs.rasterizer.vs_output_attributes[attrib].map_w.Value(),
        };

        for (u32 comp = 0; comp < 4; ++comp) {
            const std::size_t semantic = static_cast<std::size_t>(semantics[comp]);
            if (semantic < 24) {
                semantic_maps[semantic] = {attrib, comp};
            } else if (semantic != Pica::RasterizerRegs::VSOutputAttributes::INVALID) {
                LOG_ERROR(Render_OpenGL, "Invalid/unknown semantic id: {}", semantic);
            }
        }
    }
}

} // namespace VideoCore
