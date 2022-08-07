// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <type_traits>
#include <glm/glm.hpp>
#include "common/hash.h"
#include "video_core/regs.h"
#include "video_core/shader/shader.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

/// Structure that the hardware rendered vertices are composed of
struct HardwareVertex {
    HardwareVertex() = default;
    HardwareVertex(const Pica::Shader::OutputVertex& v, bool flip_quaternion) {
        position[0] = v.pos.x.ToFloat32();
        position[1] = v.pos.y.ToFloat32();
        position[2] = v.pos.z.ToFloat32();
        position[3] = v.pos.w.ToFloat32();
        color[0] = v.color.x.ToFloat32();
        color[1] = v.color.y.ToFloat32();
        color[2] = v.color.z.ToFloat32();
        color[3] = v.color.w.ToFloat32();
        tex_coord0[0] = v.tc0.x.ToFloat32();
        tex_coord0[1] = v.tc0.y.ToFloat32();
        tex_coord1[0] = v.tc1.x.ToFloat32();
        tex_coord1[1] = v.tc1.y.ToFloat32();
        tex_coord2[0] = v.tc2.x.ToFloat32();
        tex_coord2[1] = v.tc2.y.ToFloat32();
        tex_coord0_w = v.tc0_w.ToFloat32();
        normquat[0] = v.quat.x.ToFloat32();
        normquat[1] = v.quat.y.ToFloat32();
        normquat[2] = v.quat.z.ToFloat32();
        normquat[3] = v.quat.w.ToFloat32();
        view[0] = v.view.x.ToFloat32();
        view[1] = v.view.y.ToFloat32();
        view[2] = v.view.z.ToFloat32();

        if (flip_quaternion) {
            normquat = -normquat;
        }
    }

    glm::vec4 position;
    glm::vec4 color;
    glm::vec2 tex_coord0;
    glm::vec2 tex_coord1;
    glm::vec2 tex_coord2;
    float tex_coord0_w;
    glm::vec4 normquat;
    glm::vec3 view;
};

/**
 * Vertex structure that the drawn screen rectangles are composed of.
 */
struct ScreenRectVertex {
    ScreenRectVertex() = default;
    ScreenRectVertex(float x, float y, float u, float v, float s) {
        position.x = x;
        position.y = y;
        tex_coord.x = u;
        tex_coord.y = v;
        tex_coord.z = s;
    }

    glm::vec2 position;
    glm::vec3 tex_coord;
};

} // namespace Vulkan
