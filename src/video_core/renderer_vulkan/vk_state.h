// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include "video_core/renderer_vulkan/vk_texture.h"
#include "video_core/renderer_vulkan/vk_pipeline.h"

namespace Vulkan {

enum class DirtyState {
    All,
    Framebuffer,
    Pipeline,
    Texture,
    Sampler,
    TexelBuffer,
    ImageTexture,
    Depth,
    Stencil,
    LogicOp,
    Viewport,
    Scissor,
    CullMode,
    VertexBuffer,
    Uniform
};

enum class UniformID {
    Pica = 0,
    Shader = 1
};

enum class TextureID {
    Tex0 = 0,
    Tex1 = 1,
    Tex2 = 2,
    TexCube = 3
};

enum class TexelBufferID {
    LF = 0,
    RG = 1,
    RGBA = 2
};

/// Tracks global Vulkan state
class VulkanState {
public:
    VulkanState() = default;
    ~VulkanState() = default;

    /// Initialize object to its initial state
    void Create();

    /// Configure drawing state
    void SetVertexBuffer(VKBuffer* buffer, vk::DeviceSize offset);
    void SetFramebuffer(VKFramebuffer* framebuffer);
    void SetPipeline(const VKPipeline* pipeline);

    /// Configure shader resources
    void SetUniformBuffer(UniformID id, VKBuffer* buffer, u32 offset, u32 size);
    void SetTexture(TextureID id, VKTexture* texture);
    void SetTexelBuffer(TexelBufferID id, VKBuffer* buffer);
    void SetImageTexture(VKTexture* image);
    void UnbindTexture(VKTexture* image);

    /// Apply all dirty state to the current Vulkan command buffer
    void Apply();

private:
    // Stage which should be applied
    DirtyState dirty_flags;

    // Input assembly
    VKBuffer* vertex_buffer = nullptr;
    vk::DeviceSize vertex_buffer_offset = 0;

    // Pipeline state
    const VKPipeline* pipeline = nullptr;

    // Shader bindings. These describe which resources
    // we have bound to the pipeline and at which
    // bind points. When the state is applied the
    // descriptor sets are updated with the new
    // resources
    struct
    {
        std::array<vk::DescriptorBufferInfo, 2> ubo;
        std::array<vk::DescriptorImageInfo, 4> texture;
        std::array<vk::DescriptorBufferInfo, 3> lut;
    } bindings = {};

    std::array<vk::DescriptorSet, 3> descriptor_sets = {};

    // Rasterization
    vk::Viewport viewport = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
    vk::CullModeFlags cull_mode = vk::CullModeFlagBits::eNone;
    vk::Rect2D scissor = {{0, 0}, {1, 1}};
    VKTexture dummy_texture;

    // Framebuffer
    VKFramebuffer* framebuffer = nullptr;
    vk::RenderPass current_render_pass = VK_NULL_HANDLE;
    vk::Rect2D framebuffer_render_area = {};
    vk::ColorComponentFlags color_mask;

    // Depth
    bool depth_enabled;
    vk::CompareOp test_func;

    // Stencil
    bool stencil_enabled;
    vk::StencilFaceFlags face_mask;
    vk::StencilOp fail_op, pass_op;
    vk::StencilOp depth_fail_op;
    vk::CompareOp compare_op;

    vk::LogicOp logic_op;
    std::array<bool, 2> clip_distance;

};

extern std::unique_ptr<VulkanState> g_vk_state;

} // namespace Vulkan
