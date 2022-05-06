// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include "video_core/renderer_vulkan/vk_texture.h"

namespace Vulkan {

enum class DirtyState {
    None,
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
    IndexBuffer,
    Uniform,
    All
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
    void SetViewport(vk::Viewport viewport);
    void SetScissor(vk::Rect2D scissor);

    /// Rendering
    void SetAttachments(VKTexture* color, VKTexture* depth_stencil);
    void SetRenderArea(vk::Rect2D render_area);
    void BeginRendering();
    void EndRendering();

    /// Configure shader resources
    void SetUniformBuffer(UniformID id, VKBuffer* buffer, u32 offset, u32 size);
    void SetTexture(TextureID id, VKTexture* texture);
    void SetTexelBuffer(TexelBufferID id, VKBuffer* buffer);
    void UnbindTexture(VKTexture* image);

    /// Apply all dirty state to the current Vulkan command buffer
    void UpdateDescriptorSet();
    void Apply();

private:
    // Stage which should be applied
    DirtyState dirty_flags;
    bool rendering = false;

    // Input assembly
    VKBuffer* vertex_buffer = nullptr, * index_buffer = nullptr;
    vk::DeviceSize vertex_offset = 0, index_offset = 0;

    // Shader bindings. These describe which resources
    // we have bound to the pipeline and at which
    // bind points. When the state is applied the
    // descriptor sets are updated with the new
    // resources
    struct
    {
        std::array<vk::DescriptorBufferInfo, 2> ubo;
        std::array<bool, 2> ubo_update;
        std::array<vk::DescriptorImageInfo, 4> texture;
        std::array<bool, 4> texture_update;
        std::array<vk::DescriptorBufferInfo, 3> lut;
        std::array<bool, 3> lut_update;
    } bindings = {};
    std::vector<vk::UniqueDescriptorSet> descriptor_sets = {};
    vk::UniqueDescriptorPool desc_pool;

    // Rasterization
    vk::Viewport viewport = { 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f };
    vk::CullModeFlags cull_mode = vk::CullModeFlagBits::eNone;
    vk::Rect2D scissor = { {0, 0}, {1, 1} };
    VKTexture dummy_texture;

    // Render attachments
    VKTexture* color_attachment = nullptr, * depth_attachment = nullptr;
    vk::Rect2D render_area = {};
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
