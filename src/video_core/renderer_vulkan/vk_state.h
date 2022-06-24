// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <variant>
#include <optional>
#include <bitset>
#include "video_core/regs.h"
#include "video_core/renderer_vulkan/vk_shader_state.h"
#include "video_core/renderer_vulkan/vk_pipeline_builder.h"
#include "video_core/renderer_vulkan/vk_texture.h"

namespace Vulkan {

constexpr u32 DESCRIPTOR_SET_COUNT = 4;

struct DrawInfo {
    glm::mat4 modelview;
    glm::vec4 i_resolution;
    glm::vec4 o_resolution;
    int layer;
};

class DescriptorUpdater {
public:
    DescriptorUpdater() { Reset(); }
    ~DescriptorUpdater() = default;

    void Reset();
    void Update();

    void PushTextureArrayUpdate(vk::DescriptorSet, u32 biding, vk::Sampler sampler,
                                std::span<vk::ImageView> views);
    void PushCombinedImageSamplerUpdate(vk::DescriptorSet set, u32 binding,
                                        vk::Sampler sampler, vk::ImageView view);
    void PushBufferUpdate(vk::DescriptorSet set, u32 binding,
                          vk::DescriptorType buffer_type, u32 offset, u32 size,
                          vk::Buffer buffer, const vk::BufferView& view = VK_NULL_HANDLE);

private:
    static constexpr u32 MAX_DESCRIPTORS = 10;
    static constexpr u32 MAX_UPDATES = 20;
    struct Descriptor {
        vk::DescriptorImageInfo image_info;
        vk::DescriptorBufferInfo buffer_info;
    };

    std::array<vk::WriteDescriptorSet, MAX_DESCRIPTORS> writes;
    std::array<vk::DescriptorImageInfo, MAX_UPDATES> image_infos;
    std::array<vk::DescriptorBufferInfo, MAX_UPDATES> buffer_infos;
    u32 image_count{0}, buffer_count{0}, write_count{0};
};

class VKSwapChain;

/// Tracks global Vulkan state
class VulkanState {
public:
    VulkanState(const std::shared_ptr<VKSwapChain>& swapchain);
    ~VulkanState();

    /// Initialize object to its initial state
    static void Create(const std::shared_ptr<VKSwapChain>& swapchain);
    static VulkanState& Get();

    /// Query state
    bool DepthTestEnabled() const { return depth_enabled && depth_writes; }
    bool StencilTestEnabled() const { return stencil_enabled && stencil_writes; }

    /// Configure drawing state
    void SetVertexBuffer(const VKBuffer& buffer, vk::DeviceSize offset);
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
    void SetColorMask(vk::ColorComponentFlags mask);
    void SetBlendEnable(bool enable);
    void SetBlendCostants(float red, float green, float blue, float alpha);
    void SetBlendOp(vk::BlendOp rgb_op, vk::BlendOp alpha_op, vk::BlendFactor src_color, vk::BlendFactor dst_color,
                    vk::BlendFactor src_alpha, vk::BlendFactor dst_alpha);

    /// Rendering
    void BeginRendering(VKTexture* color, VKTexture* depth, bool update_pipeline_formats = false,
                        vk::ClearColorValue color_clear = {},
                        vk::AttachmentLoadOp color_load_op = vk::AttachmentLoadOp::eLoad,
                        vk::AttachmentStoreOp color_store_op = vk::AttachmentStoreOp::eStore,
                        vk::ClearDepthStencilValue depth_clear = {},
                        vk::AttachmentLoadOp depth_load_op = vk::AttachmentLoadOp::eLoad,
                        vk::AttachmentStoreOp depth_store_op = vk::AttachmentStoreOp::eStore,
                        vk::AttachmentLoadOp stencil_load_op = vk::AttachmentLoadOp::eDontCare,
                        vk::AttachmentStoreOp stencil_store_op = vk::AttachmentStoreOp::eDontCare);
    void EndRendering();

    /// Configure shader resources
    void SetUniformBuffer(u32 binding, u32 offset, u32 size, const VKBuffer& buffer);
    void SetTexture(u32 binding,  const VKTexture& texture);
    void SetTexelBuffer(u32 binding, u32 offset, u32 size, const VKBuffer& buffer, u32 view_index);
    void SetPresentTextures(vk::ImageView view0, vk::ImageView view1, vk::ImageView view2);
    void SetPresentData(DrawInfo data);
    void SetPlaceholderColor(u8 red, u8 green, u8 blue, u8 alpha);
    void UnbindTexture(const VKTexture& image);
    void UnbindTexture(u32 unit);

    /// Apply all dirty state to the current Vulkan command buffer
    void InitDescriptorSets();
    void ApplyRenderState(const Pica::Regs& config);
    void ApplyPresentState();
    void ApplyCommonState(bool extended);

private:
    void BuildDescriptorLayouts();
    void ConfigureRenderPipeline();
    void ConfigurePresentPipeline();

private:
    // Render targets
    std::shared_ptr<VKSwapChain> swapchain;
    bool rendering{false};
    vk::ImageView present_view;
    std::array<vk::ImageView, 4> render_views;
    vk::Sampler render_sampler, present_sampler;
    VKTexture placeholder;

    // Render state
    bool descriptors_dirty{};
    DescriptorUpdater updater;
    std::array<vk::DescriptorSetLayout, DESCRIPTOR_SET_COUNT> descriptor_layouts;
    std::array<vk::DescriptorSet, DESCRIPTOR_SET_COUNT> descriptor_sets;

    // Pipeline caches
    PipelineCacheKey render_pipeline_key{};
    PipelineBuilder render_pipeline_builder, present_pipeline_builder;
    vk::PipelineLayout render_pipeline_layout, present_pipeline_layout;
    std::unordered_map<PipelineCacheKey, vk::Pipeline> render_pipelines;
    vk::Pipeline present_pipeline;

    // Shader caches
    vk::ShaderModule render_vertex_shader, present_vertex_shader, present_fragment_shader;
    std::unordered_map<PicaFSConfig, vk::ShaderModule> render_fragment_shaders;

    // Dynamic state
    enum DynamicStateFlags : u32 {
        Viewport,
        Scissor,
        LineWidth,
        DepthTest,
        DepthWrite,
        StencilTest,
        StencilMask,
        ColorWrite,
        CullMode,
        BlendConstants,
        FrontFace,
    };

    std::bitset<16> dirty_flags;
    u32 stencil_write_mask{}, stencil_input_mask{}, stencil_ref{};
    bool depth_enabled{}, depth_writes{}, stencil_enabled{}, stencil_writes{};
    vk::StencilOp fail_op, pass_op, depth_fail_op;
    vk::CompareOp depth_op, stencil_op;

    vk::Viewport viewport{0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
    vk::CullModeFlags cull_mode{};
    vk::FrontFace front_face{};
    vk::Rect2D scissor{};
    vk::LogicOp logic_op{};
    std::array<float, 4> blend_constants{};
};

extern std::unique_ptr<VulkanState> s_vulkan_state;

} // namespace Vulkan
