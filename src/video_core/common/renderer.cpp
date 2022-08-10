// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <glm/gtc/matrix_transform.hpp>
#include "common/assert.h"
#include "common/logging/log.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/framebuffer_layout.h"
#include "core/hw/gpu.h"
#include "core/hw/hw.h"
#include "core/hw/lcd.h"
#include "core/settings.h"
#include "video_core/common/renderer.h"
#include "video_core/common/rasterizer.h"
#include "video_core/renderer_vulkan/vk_backend.h"
#include "video_core/video_core.h"

namespace VideoCore {

static std::string vertex_shader_source = R"(
#version 450 core
layout (location = 0) in vec2 vert_position;
layout (location = 1) in vec2 vert_tex_coord;
layout (location = 0) out vec2 frag_tex_coord;

layout (std140, push_constant) uniform PresentUniformData {
    mat4 modelview_matrix;
    vec4 i_resolution;
    vec4 o_resolution;
    int screen_id;
    int layer;
    int reverse_interlaced;
};

void main() {
    vec4 position = vec4(vert_position, 0.0, 1.0) * modelview_matrix;
    gl_Position = vec4(position.x, -position.y, 0.0, 1.0);
    frag_tex_coord = vert_tex_coord;
}
)";

static std::string fragment_shader_source = R"(
layout (location = 0) in vec2 frag_tex_coord;
layout (location = 0) out vec4 color;
layout (set = 0, binding = 0) uniform texture2D top_screen;

layout (std140, push_constant) uniform PresentUniformData {
    mat4 modelview_matrix;
    vec4 i_resolution;
    vec4 o_resolution;
    int screen_id;
    int layer;
    int reverse_interlaced;
};

void main() {
    color = texture(top_screen, frag_tex_coord);
}
)";

static std::string fragment_shader_anaglyph_source = R"(

// Anaglyph Red-Cyan shader based on Dubois algorithm
// Constants taken from the paper:
// "Conversion of a Stereo Pair to Anaglyph with
// the Least-Squares Projection Method"
// Eric Dubois, March 2009
const mat3 l = mat3(0.437, 0.449, 0.164,
              -0.062,-0.062,-0.024,
              -0.048,-0.050,-0.017);
const mat3 r = mat3(-0.011,-0.032,-0.007,
               0.377, 0.761, 0.009,
              -0.026,-0.093, 1.234);

layout (location = 0) in vec2 frag_tex_coord;
layout (location = 0) out vec4 color;
layout (set = 0, binding = 0) uniform sampler2D top_screen;
layout (set = 0, binding = 1) uniform sampler2D top_screen_r;

void main() {
    vec4 color_tex_l = texture(top_screen, frag_tex_coord);
    vec4 color_tex_r = texture(top_screen_r, frag_tex_coord);
    color = vec4(color_tex_l.rgb * l + color_tex_r.rgb * r, color_tex_l.a);
}
)";

static std::string fragment_shader_interlaced_source = R"(

layout (location = 0) in vec2 frag_tex_coord;
layout (location = 0) out vec4 color;

layout (std140, push_constant) uniform PresentUniformData {
    mat4 modelview_matrix;
    vec4 i_resolution;
    vec4 o_resolution;
    int layer;
    int reverse_interlaced;
};

layout (set = 0, binding = 0) uniform sampler2D top_screen;
layout (set = 0, binding = 1) uniform sampler2D top_screen_r;

void main() {
    float screen_row = o_resolution.x * frag_tex_coord.x;
    if (int(screen_row) % 2 == reverse_interlaced) {
        color = texture(top_screen, frag_tex_coord);
    } else {
        color = texture(top_screen_r, frag_tex_coord);
    }
}
)";

constexpr VertexLayout ScreenRectVertex::GetVertexLayout() {
    VertexLayout layout{};
    layout.attribute_count = 2;
    layout.binding_count = 1;

    // Define binding
    layout.bindings[0].binding.Assign(0);
    layout.bindings[0].fixed.Assign(0);
    layout.bindings[0].stride.Assign(sizeof(ScreenRectVertex));

    // Define the attributes
    for (u32 loc = 0; loc < 2; loc++) {
        layout.attributes[loc].binding.Assign(0);
        layout.attributes[loc].location.Assign(loc);
        layout.attributes[loc].offset.Assign(loc * sizeof(glm::vec2));
        layout.attributes[loc].size.Assign(2);
        layout.attributes[loc].type.Assign(AttribType::Float);
    }

    return layout;
}

// Renderer pipeline layout
static constexpr PipelineLayoutInfo RENDERER_PIPELINE_INFO = {
    .group_count = 2,
    .binding_groups = {
        BindingGroup{
            BindingType::Texture, // Top screen
            BindingType::Texture // Top screen stereo pair
        },
        BindingGroup{
            BindingType::Sampler
        }
    },
    .push_constant_block_size = sizeof(PresentUniformData)
};

DisplayRenderer::DisplayRenderer(Frontend::EmuWindow& window) : render_window(window) {
    //window.mailbox = nullptr;
    backend = std::make_unique<Vulkan::Backend>(window);
    rasterizer = std::make_unique<VideoCore::Rasterizer>(window, backend);

    // Create vertex buffer for the screen rectangle
    const BufferInfo vertex_info = {
        .capacity = sizeof(ScreenRectVertex) * 10,
        .usage = BufferUsage::Vertex
    };

    vertex_buffer = backend->CreateBuffer(vertex_info);

    const std::array fragment_shaders = {&fragment_shader_source,
                                         &fragment_shader_anaglyph_source,
                                         &fragment_shader_interlaced_source};

    PipelineInfo present_pipeline_info = {
        .vertex_layout = ScreenRectVertex::GetVertexLayout(),
        .layout = RENDERER_PIPELINE_INFO,
        .color_attachment = TextureFormat::PresentColor,
        .depth_attachment = TextureFormat::Undefined
    };

    // Set topology to strip
    present_pipeline_info.rasterization.topology.Assign(Pica::TriangleTopology::Strip);


    // Create vertex and fragment shaders
    vertex_shader = backend->CreateShader(ShaderStage::Vertex, "Present vertex shader",
                                          vertex_shader_source);
    for (int i = 0; i < PRESENT_PIPELINES; i++) {
        const std::string name = fmt::format("Present shader {:d}", i);
        present_shaders[i] = backend->CreateShader(ShaderStage::Fragment, name, *fragment_shaders[i]);

        // Create associated pipeline
        present_pipeline_info.shaders[0] = vertex_shader;
        present_pipeline_info.shaders[1] = present_shaders[i];
        present_pipelines[i] = backend->CreatePipeline(PipelineType::Graphics, present_pipeline_info);
    }
}

void DisplayRenderer::PrepareRendertarget() {
    for (int i = 0; i < 3; i++) {
        int fb_id = i == 2 ? 1 : 0;
        const auto& framebuffer = GPU::g_regs.framebuffer_config[fb_id];

        // Main LCD (0): 0x1ED02204, Sub LCD (1): 0x1ED02A04
        u32 lcd_color_addr = (fb_id == 0) ? LCD_REG_INDEX(color_fill_top) : LCD_REG_INDEX(color_fill_bottom);
        lcd_color_addr = HW::VADDR_LCD + 4 * lcd_color_addr;
        LCD::Regs::ColorFill color_fill = {0};
        LCD::Read(color_fill.raw, lcd_color_addr);

        if (color_fill.is_enabled) {
            LoadColorToActiveTexture(color_fill.color_r, color_fill.color_g, color_fill.color_b, screen_infos[i]);
        } else {
            const TextureHandle& texture = screen_infos[i].texture;
            u32 fwidth = framebuffer.width;
            u32 fheight = framebuffer.height;

            if (texture->GetWidth() != fwidth || texture->GetHeight() != fheight ||
                screen_infos[i].format != framebuffer.color_format) {
                // Reallocate texture if the framebuffer size has changed.
                // This is expected to not happen very often and hence should not be a
                // performance problem.
                ConfigureFramebufferTexture(screen_infos[i], framebuffer);
            }

            LoadFBToScreenInfo(framebuffer, screen_infos[i], i == 1);
        }
    }
}

void DisplayRenderer::LoadFBToScreenInfo(const GPU::Regs::FramebufferConfig& framebuffer,
                                         ScreenInfo& screen_info, bool right_eye) {

    if (framebuffer.address_right1 == 0 || framebuffer.address_right2 == 0) {
        right_eye = false;
    }

    const PAddr framebuffer_addr = framebuffer.active_fb == 0
            ? (!right_eye ? framebuffer.address_left1 : framebuffer.address_right1)
            : (!right_eye ? framebuffer.address_left2 : framebuffer.address_right2);

    LOG_TRACE(Render_Vulkan, "0x{:08x} bytes from 0x{:08x}({}x{}), fmt {:x}",
              framebuffer.stride * framebuffer.height, framebuffer_addr, framebuffer.width.Value(),
              framebuffer.height.Value(), framebuffer.format);

    int bpp = GPU::Regs::BytesPerPixel(framebuffer.color_format);
    std::size_t pixel_stride = framebuffer.stride / bpp;

    // OpenGL only supports specifying a stride in units of pixels, not bytes, unfortunately
    ASSERT(pixel_stride * bpp == framebuffer.stride);

    // Ensure no bad interactions with GL_UNPACK_ALIGNMENT, which by default
    // only allows rows to have a memory alignement of 4.
    ASSERT(pixel_stride % 4 == 0);

    if (!rasterizer->AccelerateDisplay(framebuffer, framebuffer_addr, static_cast<u32>(pixel_stride), screen_info)) {
        ASSERT(false);
        // Reset the screen info's display texture to its own permanent texture
        screen_info.display_texture = screen_info.texture;
        screen_info.display_texcoords = Common::Rectangle<f32>{0.f, 0.f, 1.f, 1.f};

        rasterizer->FlushRegion(framebuffer_addr, framebuffer.stride * framebuffer.height);

        //const u8* data_ptr = VideoCore::g_memory->GetPhysicalPointer(framebuffer_addr);
        //const u32 data_size = screen_info.texture->GetWidth() * screen_info.texture->GetHeight() *
        //auto framebuffer_data = std::span<const u8>{data_ptr, screen_info.texture.GetSize()};

        //Rect2D region{0, 0, framebuffer.width, framebuffer.height};
        //screen_info.texture->Upload(region, pixel_stride, framebuffer_data);
    }
}

void DisplayRenderer::LoadColorToActiveTexture(u8 color_r, u8 color_g, u8 color_b, const ScreenInfo& screen) {
    /*state.texture_units[0].texture_2d = texture.resource.handle;
    state.Apply();

    glActiveTexture(GL_TEXTURE0);
    u8 framebuffer_data[3] = {color_r, color_g, color_b};

    // Update existing texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, framebuffer_data);

    state.texture_units[0].texture_2d = 0;
    state.Apply();*/
}

void DisplayRenderer::ConfigureFramebufferTexture(ScreenInfo& screen, const GPU::Regs::FramebufferConfig& framebuffer) {
    screen.format = framebuffer.color_format;

    auto ToTextureFormat = [&screen]() {
        switch (screen.format) {
        case GPU::Regs::PixelFormat::RGBA8:
            return TextureFormat::RGBA8;
        case GPU::Regs::PixelFormat::RGB8:
            return TextureFormat::RGB8;
        case GPU::Regs::PixelFormat::RGB565:
            return TextureFormat::RGB565;
        case GPU::Regs::PixelFormat::RGB5A1:
            return TextureFormat::RGB5A1;
        case GPU::Regs::PixelFormat::RGBA4:
            return TextureFormat::RGBA4;
        default:
            UNIMPLEMENTED();
        }
    };

    const TextureInfo texture_info = {
        .width = static_cast<u16>(framebuffer.width),
        .height = static_cast<u16>(framebuffer.height),
        .levels = 1,
        .type = TextureType::Texture2D,
        .view_type = TextureViewType::View2D,
        .format = ToTextureFormat()
    };

    screen.texture = backend->CreateTexture(texture_info);
}

void DisplayRenderer::ReloadPresentPipeline() {
    const auto& render_3d = Settings::values.render_3d;

    // Update current pipeline
    switch (render_3d) {
    case Settings::StereoRenderOption::Anaglyph:
        current_pipeline = present_pipelines[1];
        break;
    case Settings::StereoRenderOption::ReverseInterlaced:
    case Settings::StereoRenderOption::Interlaced:
        current_pipeline = present_pipelines[2];
        break;
    default:
        current_pipeline = present_pipelines[0];
    }

    // Update uniform data
    uniform_data.reverse_interlaced = (render_3d == Settings::StereoRenderOption::ReverseInterlaced);
}

void DisplayRenderer::DrawSingleScreen(u32 screen, bool rotate, float x, float y, float w, float h) {
    const ScreenInfo& screen_info = screen_infos[screen];
    const auto& texcoords = screen_info.display_texcoords;

    // Clear the swapchain framebuffer
    FramebufferHandle display = backend->GetWindowFramebuffer();
    display->DoClear(clear_color, 0.f, 0);

    // Update viewport and scissor
    const auto& color_surface = display->GetColorAttachment();
    current_pipeline->SetViewport(0.f, 0.f, color_surface->GetWidth(), color_surface->GetHeight());
    current_pipeline->SetScissor(0, 0, color_surface->GetWidth(), color_surface->GetHeight());

    std::array<ScreenRectVertex, 4> vertices;
    if (rotate) {
        vertices = {
            ScreenRectVertex{x, y, texcoords.bottom, texcoords.left},
            ScreenRectVertex{x + w, y, texcoords.bottom, texcoords.right},
            ScreenRectVertex{x, y + h, texcoords.top, texcoords.left},
            ScreenRectVertex{x + w, y + h, texcoords.top, texcoords.right}
        };
    } else {
        vertices = {
            ScreenRectVertex{x, y, texcoords.bottom, texcoords.right},
            ScreenRectVertex{x + w, y, texcoords.top, texcoords.right},
            ScreenRectVertex{x, y + h, texcoords.bottom, texcoords.left},
            ScreenRectVertex{x + w, y + h, texcoords.top, texcoords.left}
        };
    }

    const u32 size = sizeof(ScreenRectVertex) * vertices.size();
    const u32 mapped_offset = vertex_buffer->GetCurrentOffset();
    auto vertex_data = vertex_buffer->Map(size);

    // Copy vertex data
    std::memcpy(vertex_data.data(), vertices.data(), size);
    vertex_buffer->Commit(size);

    // As this is the "DrawSingleScreenRotated" function, the output resolution dimensions have been
    // swapped. If a non-rotated draw-screen function were to be added for book-mode games, those
    // should probably be set to the standard (w, h, 1.0 / w, 1.0 / h) ordering.
    const u16 scale_factor = VideoCore::GetResolutionScaleFactor();
    const u32 width = screen_info.texture->GetWidth();
    const u32 height = screen_info.texture->GetHeight();

    uniform_data.i_resolution = glm::vec4{width * scale_factor, height * scale_factor,
                                       1.0f / (width * scale_factor),
                                       1.0f / (height * scale_factor)};
    uniform_data.o_resolution = glm::vec4{h, w, 1.0f / h, 1.0f / w};

    // Upload uniform data
    current_pipeline->BindPushConstant(uniform_data.AsBytes());

    // Bind the vertex buffer and draw
    const std::array offsets = {mapped_offset};
    backend->BindVertexBuffer(vertex_buffer, offsets);
    backend->Draw(current_pipeline, FramebufferHandle{}, 0, vertices.size());
}

void DisplayRenderer::DrawScreens(bool flipped) {
    const auto& layout = render_window.GetFramebufferLayout();
    if (VideoCore::g_renderer_bg_color_update_requested.exchange(false)) {
        // Update background color before drawing
        clear_color = Common::Vec4f{Settings::values.bg_red, Settings::values.bg_green,
                                    Settings::values.bg_blue, 0.0f};
    }

    // Set the new filtering mode for the sampler
    if (VideoCore::g_renderer_sampler_update_requested.exchange(false)) {
        ReloadSampler();
    }

    // Update present pipeline before drawing
    if (VideoCore::g_renderer_shader_update_requested.exchange(false)) {
        ReloadPresentPipeline();
    }

    const auto& top_screen = layout.top_screen;
    //const auto& bottom_screen = layout.bottom_screen;

    // Set projection matrix
    uniform_data.modelview = glm::transpose(glm::ortho<float>(0.f, layout.width, layout.height, 0.0f, 0.f, 1.f));

    uniform_data.layer = 0;
    if (layout.top_screen_enabled) {
        if (Settings::values.render_3d == Settings::StereoRenderOption::Off) {
            DrawSingleScreen(0, layout.is_rotated, top_screen.left, top_screen.top,
                             top_screen.GetWidth(), top_screen.GetHeight());
        } else if (Settings::values.render_3d == Settings::StereoRenderOption::SideBySide) {
            DrawSingleScreen(0, layout.is_rotated, top_screen.left / 2.f, top_screen.top,
                             top_screen.GetWidth() / 2.f, top_screen.GetHeight());
            uniform_data.layer = 1;
            DrawSingleScreen(1, layout.is_rotated, (top_screen.left / 2.f) + (layout.width / 2.f), top_screen.top,
                             top_screen.GetWidth() / 2.f, top_screen.GetHeight());
        } else if (Settings::values.render_3d == Settings::StereoRenderOption::CardboardVR) {
            DrawSingleScreen(0, layout.is_rotated, layout.top_screen.left, layout.top_screen.top,
                             layout.top_screen.GetWidth(), layout.top_screen.GetHeight());
            uniform_data.layer = 1;
            DrawSingleScreen(1, layout.is_rotated, layout.cardboard.top_screen_right_eye + (layout.width / 2.f),
                             layout.top_screen.top, layout.top_screen.GetWidth(), layout.top_screen.GetHeight());
        }
    }

    uniform_data.layer = 0;
    /*if (layout.bottom_screen_enabled) {
        if (layout.is_rotated) {
            if (Settings::values.render_3d == Settings::StereoRenderOption::Off) {
                DrawSingleScreenRotated(2, (float)bottom_screen.left,
                                        (float)bottom_screen.top, (float)bottom_screen.GetWidth(),
                                        (float)bottom_screen.GetHeight());
            } else if (Settings::values.render_3d == Settings::StereoRenderOption::SideBySide) {
                DrawSingleScreenRotated(
                    2, (float)bottom_screen.left / 2, (float)bottom_screen.top,
                    (float)bottom_screen.GetWidth() / 2, (float)bottom_screen.GetHeight());
                uniform_data.layer = 1;
                DrawSingleScreenRotated(
                    2, ((float)bottom_screen.left / 2) + ((float)layout.width / 2),
                    (float)bottom_screen.top, (float)bottom_screen.GetWidth() / 2,
                    (float)bottom_screen.GetHeight());
            } else if (Settings::values.render_3d == Settings::StereoRenderOption::CardboardVR) {
                DrawSingleScreenRotated(2, layout.bottom_screen.left,
                                        layout.bottom_screen.top, layout.bottom_screen.GetWidth(),
                                        layout.bottom_screen.GetHeight());
                uniform_data.layer = 1;
                DrawSingleScreenRotated(2,
                                        layout.cardboard.bottom_screen_right_eye +
                                            ((float)layout.width / 2),
                                        layout.bottom_screen.top, layout.bottom_screen.GetWidth(),
                                        layout.bottom_screen.GetHeight());
            }
        } else {
            if (Settings::values.render_3d == Settings::StereoRenderOption::Off) {
                DrawSingleScreen(2, (float)bottom_screen.left,
                                 (float)bottom_screen.top, (float)bottom_screen.GetWidth(),
                                 (float)bottom_screen.GetHeight());
            } else if (Settings::values.render_3d == Settings::StereoRenderOption::SideBySide) {
                DrawSingleScreen(2, (float)bottom_screen.left / 2,
                                 (float)bottom_screen.top, (float)bottom_screen.GetWidth() / 2,
                                 (float)bottom_screen.GetHeight());
                uniform_data.layer = 1;
                DrawSingleScreen(2,
                                 ((float)bottom_screen.left / 2) + ((float)layout.width / 2),
                                 (float)bottom_screen.top, (float)bottom_screen.GetWidth() / 2,
                                 (float)bottom_screen.GetHeight());
            } else if (Settings::values.render_3d == Settings::StereoRenderOption::CardboardVR) {
                DrawSingleScreen(2, layout.bottom_screen.left,
                                 layout.bottom_screen.top, layout.bottom_screen.GetWidth(),
                                 layout.bottom_screen.GetHeight());
                uniform_data.layer = 1;
                DrawSingleScreen(2,
                                 layout.cardboard.bottom_screen_right_eye +
                                     ((float)layout.width / 2),
                                 layout.bottom_screen.top, layout.bottom_screen.GetWidth(),
                                 layout.bottom_screen.GetHeight());
            }
        }
    }*/
}

void DisplayRenderer::SwapBuffers() {
    // Configure current framebuffer and recreate swapchain if necessary
    PrepareRendertarget();

    // Present the 3DS screens
    if (backend->BeginPresent()) {
        DrawScreens(false);
        backend->EndPresent();
    }
}

void DisplayRenderer::UpdateCurrentFramebufferLayout(bool is_portrait_mode) {
    const Layout::FramebufferLayout& layout = render_window.GetFramebufferLayout();
    render_window.UpdateCurrentFramebufferLayout(layout.width, layout.height, is_portrait_mode);
}

} // namespace Vulkan
