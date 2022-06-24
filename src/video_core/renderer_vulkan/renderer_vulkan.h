// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include "common/common_types.h"
#include "common/math_util.h"
#include "core/hw/gpu.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"
#include "video_core/renderer_vulkan/vk_state.h"

namespace Layout {
struct FramebufferLayout;
}

namespace Vulkan {

/// Structure used for storing information about the display target for each 3DS screen
struct ScreenInfo {
    Vulkan::VKTexture* display_texture;
    Common::Rectangle<float> display_texcoords;
    Vulkan::VKTexture texture;
    GPU::Regs::PixelFormat format;
};

class RendererVulkan : public VideoCore::RendererBase {
public:
    RendererVulkan(Frontend::EmuWindow& window);
    ~RendererVulkan() override = default;

    /// Initialize the renderer
    VideoCore::ResultStatus Init() override;

    /// Shutdown the renderer
    void ShutDown() override;

    bool BeginPresent();
    void EndPresent();
    void SwapBuffers() override;

    void TryPresent(int timeout_ms) override {}
    void PrepareVideoDumping() override {}
    void CleanupVideoDumping() override {}

private:
    void CreateVulkanObjects();
    void ConfigureRenderPipeline();
    void PrepareRendertarget();
    void ConfigureFramebufferTexture(ScreenInfo& screen, const GPU::Regs::FramebufferConfig& framebuffer);

    void DrawScreens(const Layout::FramebufferLayout& layout, bool flipped);
    void DrawSingleScreenRotated(u32 screen_id, float x, float y, float w, float h);
    void DrawSingleScreen(u32 screen_id, float x, float y, float w, float h);
    void DrawSingleScreenStereoRotated(const ScreenInfo& screen_info_l,
                                       const ScreenInfo& screen_info_r, float x, float y, float w, float h);
    void DrawSingleScreenStereo(const ScreenInfo& screen_info_l, const ScreenInfo& screen_info_r,
                                float x, float y, float w, float h);

    void UpdateFramerate();

    // Loads framebuffer from emulated memory into the display information structure
    void LoadFBToScreenInfo(const GPU::Regs::FramebufferConfig& framebuffer,
                            ScreenInfo& screen_info, bool right_eye);
    // Fills active OpenGL texture with the given RGB color.
    void LoadColorToActiveGLTexture(u8 color_r, u8 color_g, u8 color_b, const ScreenInfo& screen);

private:
    // Vulkan state
    DrawInfo draw_info{};
    StreamBuffer vertex_buffer;
    vk::ClearColorValue clear_color{};

    /// Display information for top and bottom screens respectively
    std::array<ScreenInfo, 3> screen_infos;
    std::shared_ptr<VKSwapChain> swapchain;
};

} // namespace OpenGL
