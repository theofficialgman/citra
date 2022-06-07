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
    VKTexture* image{};
    vk::ClearValue clear_color;
    vk::AttachmentLoadOp load_op{vk::AttachmentLoadOp::eLoad};
    vk::AttachmentStoreOp store_op{vk::AttachmentStoreOp::eStore};
};

constexpr u32 DESCRIPTOR_SET_LAYOUT_COUNT = 3;

class DescriptorUpdater {
    enum : u32
    {
        MAX_WRITES = 16,
        MAX_IMAGE_INFOS = 8,
        MAX_BUFFER_INFOS = 4,
        MAX_VIEWS = 4,
        MAX_SETS = 6
    };

public:
    DescriptorUpdater();
    ~DescriptorUpdater() = default;

    void Clear();
    void Update();

    template <typename T>
    T GetResource(u32 binding);

    void SetDescriptorSet(vk::DescriptorSet set);
    void AddCombinedImageSamplerDescriptorWrite(u32 binding, vk::Sampler sampler, const VKTexture& image);
    void AddBufferDescriptorWrite(u32 binding, vk::DescriptorType buffer_type, u32 offset, u32 size,
                                  const VKBuffer& buffer, const vk::BufferView& view = VK_NULL_HANDLE);
private:
    vk::DescriptorSet set;
    std::array<vk::WriteDescriptorSet, MAX_WRITES> writes;
    std::array<vk::DescriptorBufferInfo, MAX_BUFFER_INFOS> buffer_infos;
    std::array<vk::DescriptorImageInfo, MAX_IMAGE_INFOS> image_infos;
    std::array<vk::BufferView, MAX_VIEWS> views;

    u32 write_count = 0, buffer_info_count = 0;
    u32 image_info_count = 0, view_count = 0;
};

/// Tracks global Vulkan state
class VulkanState {
public:
    VulkanState();
    ~VulkanState() = default;

    /// Initialize object to its initial state
    static void Create();
    static VulkanState& Get();
    static vk::ShaderModule CompileShader(const std::string& source, vk::ShaderStageFlagBits stage);

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
    void SetFragmentShader(const Pica::Regs& config);
    void BeginRendering(Attachment color, Attachment depth_stencil);
    void EndRendering();

    /// Configure shader resources
    void SetUniformBuffer(BindingID id, u32 offset, u32 size, const VKBuffer& buffer);
    void SetTexture(BindingID id,  const VKTexture& texture);
    void SetTexelBuffer(BindingID id, u32 offset, u32 size, const VKBuffer& buffer, u32 view_index);
    void UnbindTexture(const VKTexture& image);
    void UnbindTexture(u32 index);

    /// Apply all dirty state to the current Vulkan command buffer
    void Apply();

private:
    void ConfigureDescriptorSets();
    void ConfigurePipeline();

private:
    DirtyFlags dirty_flags;
    DescriptorUpdater updater;
    VKTexture placeholder;
    vk::UniqueSampler sampler;

    // Vertex buffer
    VKBuffer* vertex_buffer{}, * index_buffer{};
    vk::DeviceSize vertex_offset, index_offset;

    // Viewport
    vk::Viewport viewport{0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
    vk::CullModeFlags cull_mode{};
    vk::FrontFace front_face{};
    vk::Rect2D scissor{};
    vk::LogicOp logic_op{};
    std::array<float, 4> blend_constants{};

    bool rendering = false;
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

template <typename T>
T DescriptorUpdater::GetResource(u32 binding) {
    for (auto& write : writes) {
        if (write.dstBinding == binding) {
            if constexpr (std::is_same_v<T, vk::ImageView>) {
                return write.pImageInfo[0].imageView;
            }
            else if constexpr (std::is_same_v<T, vk::Buffer>) {
                return write.pBufferInfo[0].buffer;
            }
        }
    }

    return VK_NULL_HANDLE;
}

} // namespace Vulkan
