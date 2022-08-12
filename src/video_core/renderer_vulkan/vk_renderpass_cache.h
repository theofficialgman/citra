// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include "video_core/common/texture.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore::Vulkan {

class Instance;
class Swapchain;

class RenderpassCache {
public:
    RenderpassCache(Instance& instance);
    ~RenderpassCache();

    // Creates the renderpass used when rendering to the swapchain
    void CreatePresentRenderpass(vk::Format format);

    vk::RenderPass GetRenderpass(TextureFormat color, TextureFormat depth, bool is_clear) const;

    // Returns the special swapchain renderpass
    vk::RenderPass GetPresentRenderpass() const {
        return present_renderpass;
    }

private:
    vk::RenderPass CreateRenderPass(vk::Format color, vk::Format depth, vk::AttachmentLoadOp load_op,
                                    vk::ImageLayout initial_layout, vk::ImageLayout final_layout) const;

private:
    Instance& instance;

    // Special renderpass used for rendering to the swapchain
    vk::RenderPass present_renderpass;
    // [color_format][depth_format][is_clear_pass]
    vk::RenderPass cached_renderpasses[MAX_COLOR_FORMATS+1][MAX_DEPTH_FORMATS+1][2];
};

} // namespace VideoCore::Vulkan
