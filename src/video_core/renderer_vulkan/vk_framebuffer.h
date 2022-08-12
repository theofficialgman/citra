// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "video_core/common/framebuffer.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {
class PoolManager;
}

namespace VideoCore::Vulkan {

class Instance;
class CommandScheduler;

class Framebuffer : public VideoCore::FramebufferBase {
    friend class Backend;
public:
    Framebuffer(Instance& instance, CommandScheduler& scheduler, PoolManager& pool_manager,
                const FramebufferInfo& info, vk::RenderPass load_renderpass,
                vk::RenderPass clear_renderpass);
    ~Framebuffer() override;

    void Free() override;
    void DoClear() override;

    // Transitions the attachments to the required layout
    void PrepareAttachments();

    // Returns the vulkan framebuffer handle
    vk::Framebuffer GetHandle() const {
        return framebuffer;
    }

    // Returns the renderpass with VK_LOAD_OP_LOAD
    vk::RenderPass GetLoadRenderpass() const {
        return load_renderpass;
    }

    // Returns the renderpass with VK_LOAD_OP_CLEAR (used for optimized GPU clear)
    vk::RenderPass GetClearRenderpass() const {
        return clear_renderpass;
    }

private:
    Instance& instance;
    CommandScheduler& scheduler;
    PoolManager& pool_manager;

    // Vulkan framebuffer
    vk::Framebuffer framebuffer;
    vk::RenderPass load_renderpass, clear_renderpass;
};

} // namespace VideoCore::Vulkan
