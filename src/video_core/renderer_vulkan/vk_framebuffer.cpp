// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#define VULKAN_HPP_NO_CONSTRUCTORS
#include "video_core/renderer_vulkan/vk_framebuffer.h"
#include "video_core/renderer_vulkan/vk_texture.h"
#include "video_core/renderer_vulkan/vk_task_scheduler.h"
#include "video_core/renderer_vulkan/vk_instance.h"

namespace VideoCore::Vulkan {

inline vk::Rect2D ToVkRect2D(Rect2D rect) {
    return vk::Rect2D{
        .offset = {rect.x, rect.y},
        .extent = {rect.width, rect.height}
    };
}

Framebuffer::Framebuffer(Instance& instance, CommandScheduler& scheduler, const FramebufferInfo& info,
                         vk::RenderPass load_renderpass, vk::RenderPass clear_renderpass) :
    FramebufferBase(info), instance(instance), scheduler(scheduler), load_renderpass(load_renderpass),
    clear_renderpass(clear_renderpass) {

    const Texture* color = static_cast<const Texture*>(info.color.Get());
    const Texture* depth_stencil = static_cast<const Texture*>(info.depth_stencil.Get());

    u32 attachment_count = 0;
    std::array<vk::ImageView, 2> attachments;

    if (color) {
        attachments[attachment_count++] = color->GetView();
    }

    if (depth_stencil) {
        attachments[attachment_count++] = depth_stencil->GetView();
    }

    const Texture* valid_texture = color ? color : depth_stencil;
    const vk::FramebufferCreateInfo framebuffer_info = {
        // The load and clear renderpass are compatible according to the specification
        // so there is no need to create multiple framebuffers
        .renderPass = load_renderpass,
        .attachmentCount = attachment_count,
        .pAttachments = attachments.data(),
        .width = valid_texture->GetWidth(),
        .height = valid_texture->GetHeight(),
        .layers = 1
    };

    vk::Device device = instance.GetDevice();
    framebuffer = device.createFramebuffer(framebuffer_info);
}

Framebuffer::~Framebuffer() {
    vk::Device device = instance.GetDevice();
    device.destroyFramebuffer(framebuffer);
}

void Framebuffer::DoClear() {
    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();

    u32 clear_value_count = 0;
    std::array<vk::ClearValue, 2> clear_values{};

    if (info.color.IsValid()) {
        vk::ClearColorValue clear_color{};
        std::memcpy(clear_color.float32.data(), clear_color_value.AsArray(), sizeof(float) * 4);

        clear_values[clear_value_count++] = vk::ClearValue {
            .color = clear_color
        };
    }

    if (info.depth_stencil.IsValid()) {
        clear_values[clear_value_count++] = vk::ClearValue {
            .depthStencil = vk::ClearDepthStencilValue {
                .depth = clear_depth_value,
                .stencil = clear_stencil_value
            }
        };
    }

    const vk::RenderPassBeginInfo begin_info = {
        .renderPass = clear_renderpass,
        .framebuffer = framebuffer,
        .renderArea = ToVkRect2D(draw_rect),
        .clearValueCount = clear_value_count,
        .pClearValues = clear_values.data()
    };

    // Begin clear pass
    command_buffer.beginRenderPass(begin_info, vk::SubpassContents::eInline);
    command_buffer.endRenderPass();
}

} // namespace VideoCore::Vulkan
