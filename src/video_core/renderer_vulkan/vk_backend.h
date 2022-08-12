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
#include "video_core/renderer_vulkan/vk_renderpass_cache.h"

namespace VideoCore::Vulkan {

class Backend final : public VideoCore::BackendBase {
public:
    Backend(Frontend::EmuWindow& window);
    ~Backend();

    bool BeginPresent() override;
    void EndPresent() override;

    FramebufferHandle GetWindowFramebuffer() override;

    u64 QueryDriver(Query query) override { return 0; }

    u64 PipelineInfoHash(const PipelineInfo& info) override;

    BufferHandle CreateBuffer(BufferInfo info) override;
    FramebufferHandle CreateFramebuffer(FramebufferInfo info) override;
    TextureHandle CreateTexture(TextureInfo info) override;
    PipelineHandle CreatePipeline(PipelineType type, PipelineInfo info) override;
    SamplerHandle CreateSampler(SamplerInfo info) override;
    ShaderHandle CreateShader(ShaderStage stage, std::string_view name, std::string source) override;

    void BindVertexBuffer(BufferHandle buffer, std::span<const u64> offsets) override;
    void BindIndexBuffer(BufferHandle buffer, AttribType index_type, u64 offset) override;

    void Draw(PipelineHandle pipeline, FramebufferHandle draw_framebuffer,
              u32 base_vertex, u32 num_vertices) override;

    void DrawIndexed(PipelineHandle pipeline, FramebufferHandle draw_framebuffer,
                     u32 base_index, u32 num_indices, u32 base_vertex) override;

    void DispatchCompute(PipelineHandle pipeline, Common::Vec3<u32> groupsize,
                         Common::Vec3<u32> groups) override {}

    // Returns the vulkan instance
    inline const Instance& GetInstance() const {
        return instance;
    }

    // Returns the vulkan command buffer scheduler
    inline CommandScheduler& GetScheduler() {
        return scheduler;
    }

private:
    vk::RenderPass GetRenderPass(TextureFormat color, TextureFormat depth, bool is_clear = false) const;

    // Allocates and binds descriptor sets for the provided pipeline
    void BindDescriptorSets(PipelineHandle pipeline);

    // Begins the renderpass for the provided framebuffer
    void BeginRenderpass(FramebufferHandle framebuffer);

    void OnCommandSwitch(u32 new_slot);

private:
    Instance instance;
    CommandScheduler scheduler;
    RenderpassCache renderpass_cache;
    Swapchain swapchain;
    vk::PipelineCache pipeline_cache;

    // A cache of pipeline owners
    std::unordered_map<u64, std::unique_ptr<PipelineOwner>, Common::IdentityHash> pipeline_owners;

    // Descriptor pools
    std::array<vk::DescriptorPool, SCHEDULER_COMMAND_COUNT> descriptor_pools;
};

} // namespace Vulkan
