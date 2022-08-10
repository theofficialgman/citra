// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <unordered_map>
#include "video_core/common/backend.h"
#include "video_core/renderer_vulkan/vk_task_scheduler.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline.h"

namespace VideoCore::Vulkan {

class Texture;

constexpr u32 RENDERPASS_COUNT = (MAX_COLOR_FORMATS + 1) * (MAX_DEPTH_FORMATS + 1);

class Backend final : public VideoCore::BackendBase {
public:
    Backend(Frontend::EmuWindow& window);
    ~Backend();

    bool BeginPresent() override;
    void EndPresent() override;

    FramebufferHandle GetWindowFramebuffer() override;

    u64 QueryDriver(Query query) override;

    u64 PipelineInfoHash(const PipelineInfo& info) override;

    BufferHandle CreateBuffer(BufferInfo info) override;
    FramebufferHandle CreateFramebuffer(FramebufferInfo info) override;
    TextureHandle CreateTexture(TextureInfo info) override;
    PipelineHandle CreatePipeline(PipelineType type, PipelineInfo info) override;
    SamplerHandle CreateSampler(SamplerInfo info) override;
    ShaderHandle CreateShader(ShaderStage stage, std::string_view name, std::string source) override;

    void BindVertexBuffer(BufferHandle buffer, std::span<const u32> offsets) override;
    void BindIndexBuffer(BufferHandle buffer, AttribType index_type, u32 offset) override;

    void Draw(PipelineHandle pipeline, FramebufferHandle draw_framebuffer,
              u32 base_vertex, u32 num_vertices) override;

    void DrawIndexed(PipelineHandle pipeline, FramebufferHandle draw_framebuffer,
                     u32 base_index, u32 num_indices, u32 base_vertex) override;

    void DispatchCompute(PipelineHandle pipeline, Common::Vec3<u32> groupsize,
                         Common::Vec3<u32> groups) override;

    // Returns the vulkan instance
    inline const Instance& GetInstance() const {
        return instance;
    }

    // Returns the vulkan command buffer scheduler
    inline CommandScheduler& GetScheduler() {
        return scheduler;
    }

private:
    vk::RenderPass CreateRenderPass(vk::Format color, vk::Format depth) const;
    vk::RenderPass GetRenderPass(TextureFormat color, TextureFormat depth) const;

    // Allocates and binds descriptor sets for the provided pipeline
    void BindDescriptorSets(PipelineHandle pipeline);

private:
    Instance instance;
    Swapchain swapchain;
    CommandScheduler scheduler;

    // The formats Citra uses are limited so we can pre-create
    // all the renderpasses we will need
    std::array<vk::RenderPass, RENDERPASS_COUNT> renderpass_cache;
    vk::PipelineCache cache;

    // A cache of pipeline owners
    std::unordered_map<PipelineLayoutInfo, PipelineOwner> pipeline_owners;

    // Descriptor pools
    std::array<vk::DescriptorPool, SCHEDULER_COMMAND_COUNT> descriptor_pools;
};

} // namespace Vulkan
