// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <variant>
#include <xxhash.h>
#include "video_core/regs.h"
#include "video_core/renderer_vulkan/vk_shader_state.h"
#include "video_core/renderer_vulkan/vk_pipeline_builder.h"
#include "video_core/renderer_vulkan/vk_texture.h"

namespace Vulkan {

enum class DirtyFlags {
    None = 0,
    Framebuffer = 1,
    Pipeline = 1 << 1,
    Texture = 1 << 2,
    Sampler = 1 << 3,
    TexelBuffer = 1 << 4,
    ImageTexture = 1 << 5,
    DepthTest = 1 << 6,
    Stencil = 1 << 7,
    LogicOp = 1 << 8,
    Viewport = 1 << 9,
    Scissor = 1 << 10,
    CullMode = 1 << 11,
    VertexBuffer = 1 << 12,
    IndexBuffer = 1 << 13,
    Uniform = 1 << 14,
    FrontFace = 1 << 15,
    BlendConsts = 1 << 16,
    ColorMask = 1 << 17,
    StencilMask = 1 << 18,
    DepthWrite = 1 << 19,
    All = (1 << 20) - 1
};

enum class BindingID {
    VertexUniform = 0,
    PicaUniform = 1,
    Tex0 = 2,
    Tex1 = 3,
    Tex2 = 4,
    TexCube = 5,
    LutLF = 6,
    LutRG = 7,
    LutRGBA = 8
};

BindingID operator + (BindingID lhs, u32 rhs) {
    return static_cast<BindingID>(static_cast<u32>(lhs) + rhs);
}

struct Attachment {
    VKTexture* color{}, *depth_stencil{};
    vk::ClearColorValue clear_color;
    vk::ClearDepthStencilValue depth_color;
    vk::Rect2D render_area{-1};
};

constexpr u32 DESCRIPTOR_SET_LAYOUT_COUNT = 3;

/// Tracks global Vulkan state
class VulkanState {
public:
    VulkanState() = default;
    ~VulkanState() = default;

    /// Initialize object to its initial state
    void Create();

    /// Query state
    bool DepthTestEnabled() const { return depth_enabled && depth_writes; }
    bool StencilTestEnabled() const { return stencil_enabled && stencil_writes; }

    /// Configure drawing state
    void SetVertexBuffer(VKBuffer* buffer, vk::DeviceSize offset);
    void SetViewport(vk::Viewport viewport);
    void SetScissor(vk::Rect2D scissor);
    void SetCullMode(vk::CullModeFlags flags);
    void SetFrontFace(vk::FrontFace face);
    void SetLogicOp(vk::LogicOp logic_op);
    void SetStencilWrite(u32 mask);
    void SetStencilInput(u32 mask);
    void SetStencilTest(bool enable, vk::StencilOp fail, vk::StencilOp pass, vk::StencilOp depth_fail,
                      vk::CompareOp compare, u32 ref);
    void SetDepthWrite(bool enable);
    void SetDepthTest(bool enable, vk::CompareOp compare);
    void SetColorMask(bool red, bool green, bool blue, bool alpha);
    void SetBlendEnable(bool enable);
    void SetBlendCostants(float red, float green, float blue, float alpha);
    void SetBlendOp(vk::BlendOp rgb_op, vk::BlendOp alpha_op, vk::BlendFactor src_color, vk::BlendFactor dst_color,
                    vk::BlendFactor src_alpha, vk::BlendFactor dst_alpha);

    /// Rendering
    void PushAttachment(Attachment attachment);
    void PopAttachment();
    void SetFragmentShader(const Pica::Regs& config);
    void BeginRendering();
    void EndRendering();

    /// Configure shader resources
    void SetUniformBuffer(BindingID id, VKBuffer* buffer, u32 offset, u32 size);
    void SetTexture(BindingID id, VKTexture* texture);
    void SetTexelBuffer(BindingID id, VKBuffer* buffer, vk::Format view_format);
    void UnbindTexture(VKTexture* image);
    void UnbindTexture(u32 index);

    /// Apply all dirty state to the current Vulkan command buffer
    void Apply();

private:
    void ConfigureDescriptorSets();
    void ConfigurePipeline();
    void UpdateDescriptorSet();
    vk::ShaderModule CompileShader(const std::string& source, vk::ShaderStageFlagBits stage);

private:
    struct Binding {
        bool dirty{};
        std::variant<VKBuffer*, VKTexture*> resource{};
        vk::UniqueBufferView buffer_view{};
    };

    DirtyFlags dirty_flags;
    bool rendering = false;
    VKTexture dummy_texture;
    vk::UniqueSampler sampler;
    std::vector<Attachment> targets;

    VKBuffer* vertex_buffer{}, * index_buffer{};
    vk::DeviceSize vertex_offset, index_offset;
    std::array<Binding, 9> bindings;
    std::vector<vk::UniqueDescriptorSet> descriptor_sets;
    vk::UniqueDescriptorPool desc_pool;

    vk::Viewport viewport{0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
    vk::CullModeFlags cull_mode{};
    vk::FrontFace front_face{};
    vk::Rect2D scissor{};
    vk::LogicOp logic_op{};
    std::array<float, 4> blend_constants{};

    u32 stencil_write_mask{}, stencil_input_mask{}, stencil_ref{};
    bool depth_enabled{}, depth_writes{}, stencil_enabled{}, stencil_writes{};
    vk::StencilOp fail_op, pass_op, depth_fail_op;
    vk::CompareOp depth_op, stencil_op;

    // Pipeline cache
    PipelineBuilder builder;
    vk::UniqueShaderModule trivial_vertex_shader;
    vk::UniquePipelineLayout pipeline_layout;
    std::vector<vk::DescriptorSetLayout> descriptor_layouts;
    PipelineCacheKey pipeline_key;

    std::unordered_map<PicaFSConfig, vk::UniqueShaderModule> fragment_shaders;
    std::unordered_map<PipelineCacheKey, vk::UniquePipeline> pipelines;
};

} // namespace Vulkan
