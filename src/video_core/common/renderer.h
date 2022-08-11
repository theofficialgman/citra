// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <glm/glm.hpp>
#include "common/math_util.h"
#include "core/hw/gpu.h"
#include "video_core/common/pipeline.h"

namespace Frontend {
class EmuWindow;
}

namespace Layout {
struct FramebufferLayout;
}

namespace VideoCore {

class BackendBase;
class Rasterizer;

// Structure used for storing information about the display target for each 3DS screen
struct ScreenInfo {
    TextureHandle display_texture;
    TextureHandle texture;
    SamplerHandle sampler;
    Common::Rectangle<f32> display_texcoords;
    GPU::Regs::PixelFormat format;
};

// Uniform data used for presenting the 3DS screens
struct PresentUniformData {
    glm::mat4 modelview;
    glm::vec4 i_resolution;
    glm::vec4 o_resolution;
    int screen_id = 0;
    int layer = 0;
    int reverse_interlaced = 0;

    // Returns an immutable byte view of the uniform data
    auto AsBytes() const {
        return std::as_bytes(std::span{this, 1});
    }
};

static_assert(sizeof(PresentUniformData) < 256, "PresentUniformData must be below 256 bytes!");

// Vertex structure that the drawn screen rectangles are composed of.
struct ScreenRectVertex {
    ScreenRectVertex() = default;
    ScreenRectVertex(float x, float y, float u, float v) :
        position(x, y), tex_coord(u, v) {}

    // Returns the pipeline vertex layout of the vertex
    constexpr static VertexLayout GetVertexLayout();

    glm::vec2 position;
    glm::vec2 tex_coord;
};

constexpr u32 PRESENT_PIPELINES = 3;

class DisplayRenderer {
public:
    DisplayRenderer(Frontend::EmuWindow& window);
    ~DisplayRenderer() = default;

    void SwapBuffers();
    void TryPresent(int timeout_ms) {}

    float GetCurrentFPS() const {
        return m_current_fps;
    }

    int GetCurrentFrame() const {
        return m_current_frame;
    }

    Rasterizer* Rasterizer() const {
        return rasterizer.get();
    }

    Frontend::EmuWindow& GetRenderWindow() {
        return render_window;
    }

    const Frontend::EmuWindow& GetRenderWindow() const {
        return render_window;
    }

    void Sync();

    // Updates the framebuffer layout of the contained render window handle.
    void UpdateCurrentFramebufferLayout(bool is_portrait_mode = {});

private:
    void PrepareRendertarget();
    void ConfigureFramebufferTexture(ScreenInfo& screen, const GPU::Regs::FramebufferConfig& framebuffer);

    // Updates display pipeline according to the shader configuration
    void ReloadPresentPipeline();

    // Updates the sampler used for special effects
    void ReloadSampler() {}

    // Draws the emulated screens to the emulator window.
    void DrawScreens(bool flipped);

    // Draws a single texture to the emulator window, optionally rotating
    // the texture to correct for the 3DS's LCD rotation.
    void DrawSingleScreen(u32 screen, bool rotated, float x, float y, float w, float h);

    // Loads framebuffer from emulated memory into the display information structure
    void LoadFBToScreenInfo(const GPU::Regs::FramebufferConfig& framebuffer,
                            ScreenInfo& screen_info, bool right_eye);
    // Fills active OpenGL texture with the given RGB color.
    void LoadColorToActiveTexture(u8 color_r, u8 color_g, u8 color_b, const ScreenInfo& screen);

private:
    std::unique_ptr<VideoCore::Rasterizer> rasterizer;
    std::unique_ptr<BackendBase> backend;
    Frontend::EmuWindow& render_window;
    Common::Vec4f clear_color;
    f32 m_current_fps = 0.0f;
    int m_current_frame = 0;

    // Present pipelines (Normal, Anaglyph, Interlaced)
    std::array<PipelineHandle, PRESENT_PIPELINES> present_pipelines;
    std::array<ShaderHandle, PRESENT_PIPELINES> present_shaders;
    PipelineHandle current_pipeline;
    ShaderHandle vertex_shader;

    // Display information for top and bottom screens respectively
    SamplerHandle screen_sampler;
    std::array<ScreenInfo, 3> screen_infos;
    PresentUniformData uniform_data;
    BufferHandle vertex_buffer;
};

} // namespace VideoCore
