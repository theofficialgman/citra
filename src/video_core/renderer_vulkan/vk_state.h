// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <variant>
#include <optional>
#include "video_core/regs.h"
#include "video_core/renderer_vulkan/vk_shader_state.h"
#include "video_core/renderer_vulkan/vk_pipeline_builder.h"
#include "video_core/renderer_vulkan/vk_texture.h"

namespace Vulkan {

constexpr u32 DESCRIPTOR_SET_COUNT = 4;

template <typename T>
using OptRef = std::optional<std::reference_wrapper<T>>;

struct DrawInfo {
    glm::mat2x3 modelview;
    glm::vec4 i_resolution;
    glm::vec4 o_resolution;
    int layer;
};

class DescriptorUpdater {
public:
    DescriptorUpdater() { Reset(); }
    ~DescriptorUpdater() = default;

    void Reset() { update_count = 0; }
    void Update();

    void PushCombinedImageSamplerUpdate(vk::DescriptorSet set, u32 binding,
                                        vk::Sampler sampler, vk::ImageView view);
    void PushBufferUpdate(vk::DescriptorSet set, u32 binding,
                          vk::DescriptorType buffer_type, u32 offset, u32 size,
                          vk::Buffer buffer, const vk::BufferView& view = VK_NULL_HANDLE);

private:
    static constexpr u32 MAX_DESCRIPTORS = 10;
    struct Descriptor {
        vk::DescriptorImageInfo image_info;
        vk::DescriptorBufferInfo buffer_info;
        vk::BufferView buffer_view;
    };

    std::array<vk::WriteDescriptorSet, MAX_DESCRIPTORS> writes;
    std::array<Descriptor, MAX_DESCRIPTORS> update_queue;
    u32 update_count{};
};

/// Tracks global Vulkan state
class VulkanState {
public:
    VulkanState();
    ~VulkanState();

    /// Initialize object to its initial state
    static void Create();
    static VulkanState& Get();

    /// Configure drawing state
    void SetVertexBuffer(const VKBuffer& buffer, vk::DeviceSize offset);
    void SetColorMask(bool red, bool green, bool blue, bool alpha);
    void SetLogicOp(vk::LogicOp logic_op);
    void SetBlendEnable(bool enable);
    void SetBlendOp(vk::BlendOp rgb_op, vk::BlendOp alpha_op, vk::BlendFactor src_color, vk::BlendFactor dst_color,
                    vk::BlendFactor src_alpha, vk::BlendFactor dst_alpha);

    /// Rendering
    void BeginRendering(OptRef<VKTexture> color, OptRef<VKTexture> depth,
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
    void SetPresentTexture(const VKTexture& image);
    void SetPresentData(DrawInfo data);
    void SetPlaceholderColor(u8 red, u8 green, u8 blue, u8 alpha);
    void UnbindTexture(const VKTexture& image);
    void UnbindTexture(u32 unit);

    /// Apply all dirty state to the current Vulkan command buffer
    void InitDescriptorSets();
    void ApplyRenderState(const Pica::Regs& config);
    void ApplyPresentState();

private:
    void BuildDescriptorLayouts();
    void ConfigureRenderPipeline();
    void ConfigurePresentPipeline();

private:
    // Render targets
    bool rendering{};
    VKTexture* color_render_target{}, *depth_render_target{};
    vk::ImageView present_view;
    std::array<vk::ImageView, 4> render_views;
    DrawInfo present_data;
    vk::Sampler render_sampler, present_sampler;
    VKTexture placeholder;

    // Render state
    bool descriptors_dirty{};
    DescriptorUpdater updater;
    std::array<vk::DescriptorSetLayout, DESCRIPTOR_SET_COUNT> descriptor_layouts;
    std::array<vk::DescriptorSet, DESCRIPTOR_SET_COUNT> descriptor_sets;

    // Pipeline caches
    PipelineCacheKey render_pipeline_key;
    PipelineBuilder render_pipeline_builder, present_pipeline_builder;
    vk::PipelineLayout render_pipeline_layout, present_pipeline_layout;
    std::unordered_map<PipelineCacheKey, vk::UniquePipeline> render_pipelines;
    vk::UniquePipeline present_pipeline;

    // Shader caches
    vk::ShaderModule render_vertex_shader, present_vertex_shader, present_fragment_shader;
    std::unordered_map<PicaFSConfig, vk::UniqueShaderModule> render_fragment_shaders;
};

} // namespace Vulkan
