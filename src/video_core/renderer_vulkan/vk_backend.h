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

constexpr u32 RENDERPASS_COUNT = MAX_COLOR_FORMATS * MAX_DEPTH_FORMATS;

class Backend : public VideoCore::BackendBase {
public:
    Backend(Frontend::EmuWindow& window);
    ~Backend();

    void SwapBuffers() override;

    BufferHandle CreateBuffer(BufferInfo info) override;

    FramebufferHandle CreateFramebuffer(FramebufferInfo info) override;

    TextureHandle CreateTexture(TextureInfo info) override;

    PipelineHandle CreatePipeline(PipelineType type, PipelineInfo info) override;

    SamplerHandle CreateSampler(SamplerInfo info) override;

    void Draw(PipelineHandle pipeline, FramebufferHandle draw_framebuffer,
              BufferHandle vertex_buffer,
              u32 base_vertex, u32 num_vertices) override;

    void DrawIndexed(PipelineHandle pipeline, FramebufferHandle draw_framebuffer,
                     BufferHandle vertex_buffer, BufferHandle index_buffer,
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

private:
    Instance instance;
    Swapchain swapchain;
    CommandScheduler scheduler;

    // The formats Citra uses are limited so we can pre-create
    // all the renderpasses we will need
    std::array<vk::RenderPass, RENDERPASS_COUNT> renderpass_cache;
    vk::PipelineCache cache;

    // Pipeline layout cache
    std::unordered_map<PipelineLayoutInfo, PipelineLayout> pipeline_layouts;
};

} // namespace Vulkan
