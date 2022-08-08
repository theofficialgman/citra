// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "video_core/common/framebuffer.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore::Vulkan {

class Instance;
class CommandScheduler;

class Framebuffer : public VideoCore::FramebufferBase {
public:
    Framebuffer(Instance& instance, CommandScheduler& scheduler, const FramebufferInfo& info,
                vk::RenderPass load_renderpass, vk::RenderPass clear_renderpass);
    ~Framebuffer() override;

    void DoClear(Common::Rectangle<u32> rect, Common::Vec4f color, float depth, u8 stencil) override;

    vk::Framebuffer GetHandle() const {
        return framebuffer;
    }

    vk::RenderPass GetLoadRenderpass() const {
        return load_renderpass;
    }

private:
    Instance& instance;
    CommandScheduler& scheduler;
    vk::Framebuffer framebuffer;
    vk::RenderPass load_renderpass, clear_renderpass;
};

} // namespace VideoCore::Vulkan
