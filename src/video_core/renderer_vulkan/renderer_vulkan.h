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

namespace Frontend {

struct Frame {
    u32 width = 0, height = 0;
    bool color_reloaded = false;
    Vulkan::VKTexture* color;
    vk::UniqueFence render_fence, present_fence;
};
} // namespace Frontend

namespace Vulkan {

/// Structure used for storing information about the display target for each 3DS screen
struct ScreenInfo {
    vk::Image display_texture;
    Common::Rectangle<float> display_texcoords;
    Vulkan::VKTexture texture;
    GPU::Regs::PixelFormat format;
};

struct DrawInfo {
    glm::vec4 i_resolution;
    glm::vec4 o_resolution;
    int layer;
};

class RendererVulkan : public RendererBase {
public:
    RendererVulkan(Frontend::EmuWindow& window);
    ~RendererVulkan() override = default;

    /// Initialize the renderer
    VideoCore::ResultStatus Init() override;

    /// Shutdown the renderer
    void ShutDown() override;

    bool BeginPresent();
    void EndPresent();

private:
    void CreateVulkanObjects();
    void ConfigureRenderPipeline();
    void ReloadSampler();
    void ReloadShader();
    void PrepareRendertarget();
    void ConfigureFramebufferTexture(ScreenInfo& screen, const GPU::Regs::FramebufferConfig& framebuffer);

    void DrawScreens(const Layout::FramebufferLayout& layout, bool flipped);
    void DrawSingleScreenRotated(const ScreenInfo& screen_info, float x, float y, float w, float h);
    void DrawSingleScreen(const ScreenInfo& screen_info, float x, float y, float w, float h);
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
    VulkanState state;

    // Vulkan objects
    vk::UniqueShaderModule vertex_shader, fragment_shader;
    vk::UniqueDescriptorSet descriptor_set;
    vk::UniqueDescriptorSetLayout descriptor_layout;
    vk::UniquePipelineLayout pipeline_layout;
    vk::UniquePipeline pipeline;
    VKBuffer vertex_buffer;

    /// Display information for top and bottom screens respectively
    std::array<ScreenInfo, 3> screen_infos;
    std::unique_ptr<VKSwapChain> swapchain;
};

} // namespace OpenGL
