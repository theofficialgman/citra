// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <glad/glad.h>
#include <queue>
#include "common/assert.h"
#include "common/bit_field.h"
#include "common/logging/log.h"
#include "common/microprofile.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/dumping/backend.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/framebuffer_layout.h"
#include "core/hw/gpu.h"
#include "core/hw/hw.h"
#include "core/hw/lcd.h"
#include "core/memory.h"
#include "core/settings.h"
#include "core/tracer/recorder.h"
#include "video_core/debug_utils/debug_utils.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_vulkan/vk_state.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/renderer_vulkan/vk_task_scheduler.h"
#include "video_core/renderer_vulkan/vk_pipeline_builder.h"
#include "video_core/video_core.h"

// Include these late to avoid polluting previous headers
#ifdef _WIN32
#include <windows.h>
// ensure include order
#include <vulkan/vulkan_win32.h>
#endif

#if !defined(_WIN32) && !defined(__APPLE__)
#include <X11/Xlib.h>
#include <vulkan/vulkan_wayland.h>
#include <vulkan/vulkan_xlib.h>
#endif

namespace Vulkan {

vk::SurfaceKHR CreateSurface(const VkInstance& instance,
                             const Frontend::EmuWindow& emu_window) {
    const auto& window_info = emu_window.GetWindowInfo();
    VkSurfaceKHR unsafe_surface = nullptr;

#ifdef _WIN32
    if (window_info.type == Core::Frontend::WindowSystemType::Windows) {
        const HWND hWnd = static_cast<HWND>(window_info.render_surface);
        const VkWin32SurfaceCreateInfoKHR win32_ci{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
                                                   nullptr, 0, nullptr, hWnd};
        if (vkCreateWin32SurfaceKHR(instance, &win32_ci, nullptr, &unsafe_surface) != VK_SUCCESS) {
            LOG_ERROR(Render_Vulkan, "Failed to initialize Win32 surface");
            UNREACHABLE();
        }
    }
#endif
#if !defined(_WIN32) && !defined(__APPLE__)
    if (window_info.type == Frontend::WindowSystemType::X11) {
        const VkXlibSurfaceCreateInfoKHR xlib_ci{
            VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR, nullptr, 0,
            static_cast<Display*>(window_info.display_connection),
            reinterpret_cast<Window>(window_info.render_surface)};
        if (vkCreateXlibSurfaceKHR(instance, &xlib_ci, nullptr, &unsafe_surface) != VK_SUCCESS) {
            LOG_ERROR(Render_Vulkan, "Failed to initialize Xlib surface");
            UNREACHABLE();
        }
    }

    if (window_info.type == Frontend::WindowSystemType::Wayland) {
        const VkWaylandSurfaceCreateInfoKHR wayland_ci{
            VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR, nullptr, 0,
            static_cast<wl_display*>(window_info.display_connection),
            static_cast<wl_surface*>(window_info.render_surface)};
        if (vkCreateWaylandSurfaceKHR(instance, &wayland_ci, nullptr, &unsafe_surface) != VK_SUCCESS) {
            LOG_ERROR(Render_Vulkan, "Failed to initialize Wayland surface");
            UNREACHABLE();
        }
    }
#endif
    if (!unsafe_surface) {
        LOG_ERROR(Render_Vulkan, "Presentation not supported on this platform");
        UNREACHABLE();
    }

    return vk::SurfaceKHR(unsafe_surface);
}

std::vector<const char*> RequiredExtensions(Frontend::WindowSystemType window_type, bool enable_debug_utils) {
    std::vector<const char*> extensions;
    extensions.reserve(6);
    switch (window_type) {
    case Frontend::WindowSystemType::Headless:
        break;
#ifdef _WIN32
    case Frontend::WindowSystemType::Windows:
        extensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
        break;
#endif
#if !defined(_WIN32) && !defined(__APPLE__)
    case Frontend::WindowSystemType::X11:
        extensions.push_back(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
        break;
    case Frontend::WindowSystemType::Wayland:
        extensions.push_back(VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME);
        break;
#endif
    default:
        LOG_ERROR(Render_Vulkan, "Presentation not supported on this platform");
        break;
    }
    if (window_type != Frontend::WindowSystemType::Headless) {
        extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
    }
    if (enable_debug_utils) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    return extensions;
}

static const char vertex_shader_source[] = R"(
layout (location = 0) in vec2 vert_position;
layout (location = 1) in vec2 vert_tex_coord;
layout (location = 0) out vec2 frag_tex_coord;

// This is a truncated 3x3 matrix for 2D transformations:
// The upper-left 2x2 submatrix performs scaling/rotation/mirroring.
// The third column performs translation.
// The third row could be used for projection, which we don't need in 2D. It hence is assumed to
// implicitly be [0, 0, 1]
layout (push_constant) uniform mat3x2 modelview_matrix;

void main() {
    // Multiply input position by the rotscale part of the matrix and then manually translate by
    // the last column. This is equivalent to using a full 3x3 matrix and expanding the vector
    // to `vec3(vert_position.xy, 1.0)`
    gl_Position = vec4(mat2(modelview_matrix) * vert_position + modelview_matrix[2], 0.0, 1.0);
    frag_tex_coord = vert_tex_coord;
}
)";

static const char fragment_shader_source[] = R"(
layout (location = 0) in vec2 frag_tex_coord;
layout (location = 0) out vec4 color;

layout (push_constant) uniform DrawInfo {
    vec4 i_resolution;
    vec4 o_resolution;
    int layer;
};

uniform sampler2D color_texture;

void main() {
    color = texture(color_texture, frag_tex_coord);
}
)";

/**
 * Vertex structure that the drawn screen rectangles are composed of.
 */

struct ScreenRectVertexBase {
    ScreenRectVertexBase() = default;
    ScreenRectVertexBase(float x, float y, float u, float v) {
        position.x = x;
        position.y = y;
        tex_coord.x = u;
        tex_coord.y = v;
    }

    glm::vec2 position;
    glm::vec2 tex_coord;
};

struct ScreenRectVertex : public ScreenRectVertexBase {
    ScreenRectVertex() = default;
    ScreenRectVertex(float x, float y, float u, float v) : ScreenRectVertexBase(x, y, u, v) {};
    static constexpr auto binding_desc = vk::VertexInputBindingDescription(0, sizeof(ScreenRectVertexBase));
    static constexpr std::array<vk::VertexInputAttributeDescription, 8> attribute_desc =
    {
          vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(ScreenRectVertexBase, position)),
          vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32Sfloat, offsetof(ScreenRectVertexBase, tex_coord)),
    };
};

/**
 * Defines a 1:1 pixel ortographic projection matrix with (0,0) on the top-left
 * corner and (width, height) on the lower-bottom.
 *
 * The projection part of the matrix is trivial, hence these operations are represented
 * by a 3x2 matrix.
 *
 * @param flipped Whether the frame should be flipped upside down.
 */
static std::array<GLfloat, 3 * 2> MakeOrthographicMatrix(const float width, const float height,
                                                         bool flipped) {

    std::array<GLfloat, 3 * 2> matrix; // Laid out in column-major order

    // Last matrix row is implicitly assumed to be [0, 0, 1].
    if (flipped) {
        // clang-format off
        matrix[0] = 2.f / width; matrix[2] = 0.f;           matrix[4] = -1.f;
        matrix[1] = 0.f;         matrix[3] = 2.f / height;  matrix[5] = -1.f;
        // clang-format on
    } else {
        // clang-format off
        matrix[0] = 2.f / width; matrix[2] = 0.f;           matrix[4] = -1.f;
        matrix[1] = 0.f;         matrix[3] = -2.f / height; matrix[5] = 1.f;
        // clang-format on
    }

    return matrix;
}

RendererVulkan::RendererVulkan(Frontend::EmuWindow& window)
    : RendererBase{window} {

    window.mailbox = nullptr;
    swapchain = std::make_unique<VKSwapChain>(CreateSurface(nullptr, window));
}

MICROPROFILE_DEFINE(OpenGL_RenderFrame, "OpenGL", "Render Frame", MP_RGB(128, 128, 64));
MICROPROFILE_DEFINE(OpenGL_WaitPresent, "OpenGL", "Wait For Present", MP_RGB(128, 128, 128));

/// Swap buffers (render frame)
void RendererVulkan::SwapBuffers() {
    PrepareRendertarget();

    const auto& layout = render_window.GetFramebufferLayout();
    DrawScreens(layout, false);

    m_current_frame++;

    Core::System::GetInstance().perf_stats->EndSystemFrame();

    render_window.PollEvents();

    Core::System::GetInstance().frame_limiter.DoFrameLimiting(
        Core::System::GetInstance().CoreTiming().GetGlobalTimeUs());
    Core::System::GetInstance().perf_stats->BeginSystemFrame();

    RefreshRasterizerSetting();

    if (Pica::g_debug_context && Pica::g_debug_context->recorder) {
        Pica::g_debug_context->recorder->FrameFinished();
    }
}

void RendererVulkan::PrepareRendertarget() {
    for (int i = 0; i < 3; i++) {
        int fb_id = i == 2 ? 1 : 0;
        const auto& framebuffer = GPU::g_regs.framebuffer_config[fb_id];

        // Main LCD (0): 0x1ED02204, Sub LCD (1): 0x1ED02A04
        u32 lcd_color_addr =
            (fb_id == 0) ? LCD_REG_INDEX(color_fill_top) : LCD_REG_INDEX(color_fill_bottom);
        lcd_color_addr = HW::VADDR_LCD + 4 * lcd_color_addr;
        LCD::Regs::ColorFill color_fill = {0};
        LCD::Read(color_fill.raw, lcd_color_addr);

        if (color_fill.is_enabled) {
            LoadColorToActiveGLTexture(color_fill.color_r, color_fill.color_g, color_fill.color_b, screen_infos[i]);
        } else {
            auto extent = screen_infos[i].texture.GetArea().extent;
            auto format = screen_infos[i].format;
            if (extent.width != framebuffer.width || extent.height != framebuffer.height ||
                format != framebuffer.color_format) {
                // Reallocate texture if the framebuffer size has changed.
                // This is expected to not happen very often and hence should not be a
                // performance problem.
                ConfigureFramebufferTexture(screen_infos[i], framebuffer);
            }

            LoadFBToScreenInfo(framebuffer, screen_infos[i], i == 1);

            // Resize the texture in case the framebuffer size has changed
            //screen_infos[i].texture.width = framebuffer.width;
            //screen_infos[i].texture.height = framebuffer.height;
        }
    }
}

/**
 * Loads framebuffer from emulated memory into the active OpenGL texture.
 */
void RendererVulkan::LoadFBToScreenInfo(const GPU::Regs::FramebufferConfig& framebuffer,
                                        ScreenInfo& screen_info, bool right_eye) {

    if (framebuffer.address_right1 == 0 || framebuffer.address_right2 == 0)
        right_eye = false;

    const PAddr framebuffer_addr =
        framebuffer.active_fb == 0
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

    if (!Rasterizer()->AccelerateDisplay(framebuffer, framebuffer_addr, static_cast<u32>(pixel_stride), screen_info)) {
        // Reset the screen info's display texture to its own permanent texture
        screen_info.display_texture = screen_info.texture.GetHandle();
        screen_info.display_texcoords = Common::Rectangle<float>(0.f, 0.f, 1.f, 1.f);

        Memory::RasterizerFlushRegion(framebuffer_addr, framebuffer.stride * framebuffer.height);

        vk::Rect2D region{{0, 0}, {framebuffer.width, framebuffer.height}};
        std::span<u8> framebuffer_data(VideoCore::g_memory->GetPhysicalPointer(framebuffer_addr),
                                       screen_info.texture.GetSize());

        screen_info.texture.Upload(0, 1, pixel_stride, region, framebuffer_data);
    }
}

/**
 * Fills active OpenGL texture with the given RGB color. Since the color is solid, the texture can
 * be 1x1 but will stretch across whatever it's rendered on.
 */
void RendererVulkan::LoadColorToActiveGLTexture(u8 color_r, u8 color_g, u8 color_b, const ScreenInfo& screen) {
    /*state.texture_units[0].texture_2d = texture.resource.handle;
    state.Apply();

    glActiveTexture(GL_TEXTURE0);
    u8 framebuffer_data[3] = {color_r, color_g, color_b};

    // Update existing texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, framebuffer_data);

    state.texture_units[0].texture_2d = 0;
    state.Apply();*/
}

/**
 * Initializes the OpenGL state and creates persistent objects.
 */
void RendererVulkan::CreateVulkanObjects() {
    glClearColor(Settings::values.bg_red, Settings::values.bg_green, Settings::values.bg_blue,
                 0.0f);

    //filter_sampler.Create();
    //ReloadSampler();

    ReloadShader();

    // Generate VBO handle for drawing
    VKBuffer::Info vertex_info{
        .size = sizeof(ScreenRectVertex) * 4,
        .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
        .usage = vk::BufferUsageFlagBits::eVertexBuffer |
                 vk::BufferUsageFlagBits::eTransferDst
    };
    vertex_buffer.Create(vertex_info);
}

void RendererVulkan::ConfigureRenderPipeline() {
    // Define the descriptor sets we will be using
    vk::DescriptorSetLayoutBinding color_texture{
        0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment
    };
    vk::DescriptorSetLayoutCreateInfo color_texture_info{{}, color_texture};

    auto& device = g_vk_instace->GetDevice();
    descriptor_layout = device.createDescriptorSetLayoutUnique(color_texture_info);

    // Build the display pipeline layout
    PipelineLayoutBuilder lbuilder;
    lbuilder.AddDescriptorSet(descriptor_layout.get());
    lbuilder.AddPushConstants(vk::ShaderStageFlagBits::eVertex, 0, sizeof(glm::mat2x3));
    lbuilder.AddPushConstants(vk::ShaderStageFlagBits::eFragment, 0, sizeof(DrawInfo));
    pipeline_layout = vk::UniquePipelineLayout{lbuilder.Build()};

    std::array<vk::DynamicState, 3> dynamic_states{
        vk::DynamicState::eLineWidth, vk::DynamicState::eViewport, vk::DynamicState::eScissor,
    };

    // Build the display pipeline
    PipelineBuilder builder;
    builder.SetNoStencilState();
    builder.SetNoBlendingState();
    builder.SetNoDepthTestState();
    builder.SetNoCullRasterizationState();
    builder.SetLineWidth(1.0f);
    builder.SetPrimitiveTopology(vk::PrimitiveTopology::eTriangleList);
    builder.SetShaderStage(vk::ShaderStageFlagBits::eVertex, vertex_shader.get());
    builder.SetShaderStage(vk::ShaderStageFlagBits::eFragment, fragment_shader.get());
    builder.SetDynamicStates(dynamic_states);
    builder.SetPipelineLayout(pipeline_layout.get());

    // Configure vertex buffer
    auto attributes = ScreenRectVertex::attribute_desc;
    builder.AddVertexBuffer(0, sizeof(ScreenRectVertex), vk::VertexInputRate::eVertex, attributes);

    pipeline = vk::UniquePipeline{builder.Build()};
}

void RendererVulkan::ReloadSampler() {
    /*glSamplerParameteri(filter_sampler.handle, GL_TEXTURE_MIN_FILTER,
                        Settings::values.filter_mode ? GL_LINEAR : GL_NEAREST);
    glSamplerParameteri(filter_sampler.handle, GL_TEXTURE_MAG_FILTER,
                        Settings::values.filter_mode ? GL_LINEAR : GL_NEAREST);
    glSamplerParameteri(filter_sampler.handle, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glSamplerParameteri(filter_sampler.handle, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);*/
}

void RendererVulkan::ReloadShader() {
    // Link shaders and get variable locations
    vertex_shader = vk::UniqueShaderModule{VulkanState::CompileShader(vertex_shader_source,
                                                                      vk::ShaderStageFlagBits::eVertex)};
    fragment_shader = vk::UniqueShaderModule{VulkanState::CompileShader(fragment_shader_source,
                                                                        vk::ShaderStageFlagBits::eFragment)};
}

void RendererVulkan::ConfigureFramebufferTexture(ScreenInfo& screen, const GPU::Regs::FramebufferConfig& framebuffer) {
    GPU::Regs::PixelFormat format = framebuffer.color_format;

    VKTexture::Info texture_info{
        .width = framebuffer.width,
        .height = framebuffer.height,
        .type = vk::ImageType::e2D,
        .view_type = vk::ImageViewType::e2D,
        .usage = vk::ImageUsageFlagBits::eColorAttachment |
                 vk::ImageUsageFlagBits::eTransferDst,
        .aspect = vk::ImageAspectFlagBits::eColor,
    };

    switch (format) {
    case GPU::Regs::PixelFormat::RGBA8:
        texture_info.format = vk::Format::eR8G8B8A8Unorm;
        break;

    case GPU::Regs::PixelFormat::RGB8:
        // This pixel format uses BGR since GL_UNSIGNED_BYTE specifies byte-order, unlike every
        // specific OpenGL type used in this function using native-endian (that is, little-endian
        // mostly everywhere) for words or half-words.
        // TODO: check how those behave on big-endian processors.
        //internal_format = GL_RGB;

        // GLES Dosen't support BGR , Use RGB instead
        //texture.gl_format = GLES ? GL_RGB : GL_BGR;
        //texture.gl_type = GL_UNSIGNED_BYTE;
        texture_info.format = vk::Format::eR8G8B8Unorm;
        break;

    case GPU::Regs::PixelFormat::RGB565:
        texture_info.format = vk::Format::eR5G6B5UnormPack16;
        break;

    case GPU::Regs::PixelFormat::RGB5A1:
        texture_info.format = vk::Format::eR5G5B5A1UnormPack16;
        break;

    case GPU::Regs::PixelFormat::RGBA4:
        texture_info.format = vk::Format::eR4G4B4A4UnormPack16;
        break;

    default:
        UNIMPLEMENTED();
    }

    auto& texture = screen.texture;
    texture.Destroy();
    texture.Create(texture_info);
}

/**
 * Draws a single texture to the emulator window, rotating the texture to correct for the 3DS's LCD
 * rotation.
 */
void RendererVulkan::DrawSingleScreenRotated(const ScreenInfo& screen_info, float x, float y,
                                             float w, float h) {
    const auto& texcoords = screen_info.display_texcoords;

    const std::array<ScreenRectVertex, 4> vertices = {{
        ScreenRectVertex(x, y, texcoords.bottom, texcoords.left),
        ScreenRectVertex(x + w, y, texcoords.bottom, texcoords.right),
        ScreenRectVertex(x, y + h, texcoords.top, texcoords.left),
        ScreenRectVertex(x + w, y + h, texcoords.top, texcoords.right),
    }};

    auto command_buffer = g_vk_task_scheduler->GetCommandBuffer();

    // As this is the "DrawSingleScreenRotated" function, the output resolution dimensions have been
    // swapped. If a non-rotated draw-screen function were to be added for book-mode games, those
    // should probably be set to the standard (w, h, 1.0 / w, 1.0 / h) ordering.
    const u16 scale_factor = VideoCore::GetResolutionScaleFactor();
    auto [width, height] = screen_info.texture.GetArea().extent;

    draw_info.i_resolution = glm::vec4(width * scale_factor, height * scale_factor,
                                       1.0f / (width * scale_factor),
                                       1.0f / (height * scale_factor));
    draw_info.o_resolution = glm::vec4(h, w, 1.0f / h, 1.0f / w);
    command_buffer.pushConstants(pipeline_layout.get(), vk::ShaderStageFlagBits::eFragment,
                                 0, sizeof(DrawInfo), &draw_info);

    state.texture_units[0].texture_2d = screen_info.display_texture;
    state.texture_units[0].sampler = filter_sampler.handle;
    state.Apply();

    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices.data());
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    state.texture_units[0].texture_2d = 0;
    state.texture_units[0].sampler = 0;
    state.Apply();
}

void RendererVulkan::DrawSingleScreen(const ScreenInfo& screen_info, float x, float y, float w,
                                      float h) {
    const auto& texcoords = screen_info.display_texcoords;

    const std::array<ScreenRectVertex, 4> vertices = {{
        ScreenRectVertex(x, y, texcoords.bottom, texcoords.right),
        ScreenRectVertex(x + w, y, texcoords.top, texcoords.right),
        ScreenRectVertex(x, y + h, texcoords.bottom, texcoords.left),
        ScreenRectVertex(x + w, y + h, texcoords.top, texcoords.left),
    }};

    const u16 scale_factor = VideoCore::GetResolutionScaleFactor();
    glUniform4f(uniform_i_resolution, static_cast<float>(screen_info.texture.width * scale_factor),
                static_cast<float>(screen_info.texture.height * scale_factor),
                1.0f / static_cast<float>(screen_info.texture.width * scale_factor),
                1.0f / static_cast<float>(screen_info.texture.height * scale_factor));
    glUniform4f(uniform_o_resolution, w, h, 1.0f / w, 1.0f / h);
    state.texture_units[0].texture_2d = screen_info.display_texture;
    state.texture_units[0].sampler = filter_sampler.handle;
    state.Apply();

    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices.data());
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    state.texture_units[0].texture_2d = 0;
    state.texture_units[0].sampler = 0;
    state.Apply();
}

/**
 * Draws a single texture to the emulator window, rotating the texture to correct for the 3DS's LCD
 * rotation.
 */
void RendererVulkan::DrawSingleScreenStereoRotated(const ScreenInfo& screen_info_l,
                                                   const ScreenInfo& screen_info_r, float x,
                                                   float y, float w, float h) {
    const auto& texcoords = screen_info_l.display_texcoords;

    const std::array<ScreenRectVertex, 4> vertices = {{
        ScreenRectVertex(x, y, texcoords.bottom, texcoords.left),
        ScreenRectVertex(x + w, y, texcoords.bottom, texcoords.right),
        ScreenRectVertex(x, y + h, texcoords.top, texcoords.left),
        ScreenRectVertex(x + w, y + h, texcoords.top, texcoords.right),
    }};

    const u16 scale_factor = VideoCore::GetResolutionScaleFactor();
    glUniform4f(uniform_i_resolution,
                static_cast<float>(screen_info_l.texture.width * scale_factor),
                static_cast<float>(screen_info_l.texture.height * scale_factor),
                1.0f / static_cast<float>(screen_info_l.texture.width * scale_factor),
                1.0f / static_cast<float>(screen_info_l.texture.height * scale_factor));
    glUniform4f(uniform_o_resolution, h, w, 1.0f / h, 1.0f / w);
    state.texture_units[0].texture_2d = screen_info_l.display_texture;
    state.texture_units[1].texture_2d = screen_info_r.display_texture;
    state.texture_units[0].sampler = filter_sampler.handle;
    state.texture_units[1].sampler = filter_sampler.handle;
    state.Apply();

    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices.data());
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    state.texture_units[0].texture_2d = 0;
    state.texture_units[1].texture_2d = 0;
    state.texture_units[0].sampler = 0;
    state.texture_units[1].sampler = 0;
    state.Apply();
}

void RendererVulkan::DrawSingleScreenStereo(const ScreenInfo& screen_info_l,
                                            const ScreenInfo& screen_info_r, float x, float y,
                                            float w, float h) {
    const auto& texcoords = screen_info_l.display_texcoords;

    const std::array<ScreenRectVertex, 4> vertices = {{
        ScreenRectVertex(x, y, texcoords.bottom, texcoords.right),
        ScreenRectVertex(x + w, y, texcoords.top, texcoords.right),
        ScreenRectVertex(x, y + h, texcoords.bottom, texcoords.left),
        ScreenRectVertex(x + w, y + h, texcoords.top, texcoords.left),
    }};

    const u16 scale_factor = VideoCore::GetResolutionScaleFactor();
    glUniform4f(uniform_i_resolution,
                static_cast<float>(screen_info_l.texture.width * scale_factor),
                static_cast<float>(screen_info_l.texture.height * scale_factor),
                1.0f / static_cast<float>(screen_info_l.texture.width * scale_factor),
                1.0f / static_cast<float>(screen_info_l.texture.height * scale_factor));
    glUniform4f(uniform_o_resolution, w, h, 1.0f / w, 1.0f / h);
    state.texture_units[0].texture_2d = screen_info_l.display_texture;
    state.texture_units[1].texture_2d = screen_info_r.display_texture;
    state.texture_units[0].sampler = filter_sampler.handle;
    state.texture_units[1].sampler = filter_sampler.handle;
    state.Apply();

    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices.data());
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    state.texture_units[0].texture_2d = 0;
    state.texture_units[1].texture_2d = 0;
    state.texture_units[0].sampler = 0;
    state.texture_units[1].sampler = 0;
    state.Apply();
}

/**
 * Draws the emulated screens to the emulator window.
 */
void RendererVulkan::DrawScreens(const Layout::FramebufferLayout& layout, bool flipped) {
    if (VideoCore::g_renderer_bg_color_update_requested.exchange(false)) {
        // Update background color before drawing
        glClearColor(Settings::values.bg_red, Settings::values.bg_green, Settings::values.bg_blue, 0.0f);
    }

    if (VideoCore::g_renderer_sampler_update_requested.exchange(false)) {
        // Set the new filtering mode for the sampler
        ReloadSampler();
    }

    /*if (VideoCore::g_renderer_shader_update_requested.exchange(false)) {
        // Update fragment shader before drawing
        shader.Release();
        // Link shaders and get variable locations
        ReloadShader();
    }*/

    const auto& top_screen = layout.top_screen;
    const auto& bottom_screen = layout.bottom_screen;

    glViewport(0, 0, layout.width, layout.height);
    glClear(GL_COLOR_BUFFER_BIT);

    // Set projection matrix
    std::array<GLfloat, 3 * 2> ortho_matrix =
        MakeOrthographicMatrix((float)layout.width, (float)layout.height, flipped);
    glUniformMatrix3x2fv(uniform_modelview_matrix, 1, GL_FALSE, ortho_matrix.data());

    // Bind texture in Texture Unit 0
    glUniform1i(uniform_color_texture, 0);

    const bool stereo_single_screen =
        Settings::values.render_3d == Settings::StereoRenderOption::Anaglyph ||
        Settings::values.render_3d == Settings::StereoRenderOption::Interlaced ||
        Settings::values.render_3d == Settings::StereoRenderOption::ReverseInterlaced;

    // Bind a second texture for the right eye if in Anaglyph mode
    if (stereo_single_screen) {
        glUniform1i(uniform_color_texture_r, 1);
    }

    glUniform1i(uniform_layer, 0);
    if (layout.top_screen_enabled) {
        if (layout.is_rotated) {
            if (Settings::values.render_3d == Settings::StereoRenderOption::Off) {
                DrawSingleScreenRotated(screen_infos[0], (float)top_screen.left,
                                        (float)top_screen.top, (float)top_screen.GetWidth(),
                                        (float)top_screen.GetHeight());
            } else if (Settings::values.render_3d == Settings::StereoRenderOption::SideBySide) {
                DrawSingleScreenRotated(screen_infos[0], (float)top_screen.left / 2,
                                        (float)top_screen.top, (float)top_screen.GetWidth() / 2,
                                        (float)top_screen.GetHeight());
                glUniform1i(uniform_layer, 1);
                DrawSingleScreenRotated(screen_infos[1],
                                        ((float)top_screen.left / 2) + ((float)layout.width / 2),
                                        (float)top_screen.top, (float)top_screen.GetWidth() / 2,
                                        (float)top_screen.GetHeight());
            } else if (Settings::values.render_3d == Settings::StereoRenderOption::CardboardVR) {
                DrawSingleScreenRotated(screen_infos[0], layout.top_screen.left,
                                        layout.top_screen.top, layout.top_screen.GetWidth(),
                                        layout.top_screen.GetHeight());
                glUniform1i(uniform_layer, 1);
                DrawSingleScreenRotated(screen_infos[1],
                                        layout.cardboard.top_screen_right_eye +
                                            ((float)layout.width / 2),
                                        layout.top_screen.top, layout.top_screen.GetWidth(),
                                        layout.top_screen.GetHeight());
            } else if (stereo_single_screen) {
                DrawSingleScreenStereoRotated(
                    screen_infos[0], screen_infos[1], (float)top_screen.left, (float)top_screen.top,
                    (float)top_screen.GetWidth(), (float)top_screen.GetHeight());
            }
        } else {
            if (Settings::values.render_3d == Settings::StereoRenderOption::Off) {
                DrawSingleScreen(screen_infos[0], (float)top_screen.left, (float)top_screen.top,
                                 (float)top_screen.GetWidth(), (float)top_screen.GetHeight());
            } else if (Settings::values.render_3d == Settings::StereoRenderOption::SideBySide) {
                DrawSingleScreen(screen_infos[0], (float)top_screen.left / 2, (float)top_screen.top,
                                 (float)top_screen.GetWidth() / 2, (float)top_screen.GetHeight());
                glUniform1i(uniform_layer, 1);
                DrawSingleScreen(screen_infos[1],
                                 ((float)top_screen.left / 2) + ((float)layout.width / 2),
                                 (float)top_screen.top, (float)top_screen.GetWidth() / 2,
                                 (float)top_screen.GetHeight());
            } else if (Settings::values.render_3d == Settings::StereoRenderOption::CardboardVR) {
                DrawSingleScreen(screen_infos[0], layout.top_screen.left, layout.top_screen.top,
                                 layout.top_screen.GetWidth(), layout.top_screen.GetHeight());
                glUniform1i(uniform_layer, 1);
                DrawSingleScreen(screen_infos[1],
                                 layout.cardboard.top_screen_right_eye + ((float)layout.width / 2),
                                 layout.top_screen.top, layout.top_screen.GetWidth(),
                                 layout.top_screen.GetHeight());
            } else if (stereo_single_screen) {
                DrawSingleScreenStereo(screen_infos[0], screen_infos[1], (float)top_screen.left,
                                       (float)top_screen.top, (float)top_screen.GetWidth(),
                                       (float)top_screen.GetHeight());
            }
        }
    }
    glUniform1i(uniform_layer, 0);
    if (layout.bottom_screen_enabled) {
        if (layout.is_rotated) {
            if (Settings::values.render_3d == Settings::StereoRenderOption::Off) {
                DrawSingleScreenRotated(screen_infos[2], (float)bottom_screen.left,
                                        (float)bottom_screen.top, (float)bottom_screen.GetWidth(),
                                        (float)bottom_screen.GetHeight());
            } else if (Settings::values.render_3d == Settings::StereoRenderOption::SideBySide) {
                DrawSingleScreenRotated(
                    screen_infos[2], (float)bottom_screen.left / 2, (float)bottom_screen.top,
                    (float)bottom_screen.GetWidth() / 2, (float)bottom_screen.GetHeight());
                glUniform1i(uniform_layer, 1);
                DrawSingleScreenRotated(
                    screen_infos[2], ((float)bottom_screen.left / 2) + ((float)layout.width / 2),
                    (float)bottom_screen.top, (float)bottom_screen.GetWidth() / 2,
                    (float)bottom_screen.GetHeight());
            } else if (Settings::values.render_3d == Settings::StereoRenderOption::CardboardVR) {
                DrawSingleScreenRotated(screen_infos[2], layout.bottom_screen.left,
                                        layout.bottom_screen.top, layout.bottom_screen.GetWidth(),
                                        layout.bottom_screen.GetHeight());
                glUniform1i(uniform_layer, 1);
                DrawSingleScreenRotated(screen_infos[2],
                                        layout.cardboard.bottom_screen_right_eye +
                                            ((float)layout.width / 2),
                                        layout.bottom_screen.top, layout.bottom_screen.GetWidth(),
                                        layout.bottom_screen.GetHeight());
            } else if (stereo_single_screen) {
                DrawSingleScreenStereoRotated(screen_infos[2], screen_infos[2],
                                              (float)bottom_screen.left, (float)bottom_screen.top,
                                              (float)bottom_screen.GetWidth(),
                                              (float)bottom_screen.GetHeight());
            }
        } else {
            if (Settings::values.render_3d == Settings::StereoRenderOption::Off) {
                DrawSingleScreen(screen_infos[2], (float)bottom_screen.left,
                                 (float)bottom_screen.top, (float)bottom_screen.GetWidth(),
                                 (float)bottom_screen.GetHeight());
            } else if (Settings::values.render_3d == Settings::StereoRenderOption::SideBySide) {
                DrawSingleScreen(screen_infos[2], (float)bottom_screen.left / 2,
                                 (float)bottom_screen.top, (float)bottom_screen.GetWidth() / 2,
                                 (float)bottom_screen.GetHeight());
                glUniform1i(uniform_layer, 1);
                DrawSingleScreen(screen_infos[2],
                                 ((float)bottom_screen.left / 2) + ((float)layout.width / 2),
                                 (float)bottom_screen.top, (float)bottom_screen.GetWidth() / 2,
                                 (float)bottom_screen.GetHeight());
            } else if (Settings::values.render_3d == Settings::StereoRenderOption::CardboardVR) {
                DrawSingleScreen(screen_infos[2], layout.bottom_screen.left,
                                 layout.bottom_screen.top, layout.bottom_screen.GetWidth(),
                                 layout.bottom_screen.GetHeight());
                glUniform1i(uniform_layer, 1);
                DrawSingleScreen(screen_infos[2],
                                 layout.cardboard.bottom_screen_right_eye +
                                     ((float)layout.width / 2),
                                 layout.bottom_screen.top, layout.bottom_screen.GetWidth(),
                                 layout.bottom_screen.GetHeight());
            } else if (stereo_single_screen) {
                DrawSingleScreenStereo(screen_infos[2], screen_infos[2], (float)bottom_screen.left,
                                       (float)bottom_screen.top, (float)bottom_screen.GetWidth(),
                                       (float)bottom_screen.GetHeight());
            }
        }
    }
}

bool RendererVulkan::BeginPresent() {
    // Previous frame needs to be presented before we can acquire the swap chain.
    g_vulkan_context->WaitForPresentComplete();

    auto available = swapchain->AcquireNextImage();
    auto cmdbuffer = g_vk_task_scheduler->GetCommandBuffer();

    // Swap chain images start in undefined
    Vulkan::Texture& swap_chain_texture = m_swap_chain->GetCurrentTexture();
    swap_chain_texture.OverrideImageLayout(VK_IMAGE_LAYOUT_UNDEFINED);
    swap_chain_texture.TransitionToLayout(cmdbuffer, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    const VkClearValue clear_value = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
    const VkRenderPassBeginInfo rp = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, nullptr, m_swap_chain->GetClearRenderPass(),
        m_swap_chain->GetCurrentFramebuffer(), {{0, 0}, {swap_chain_texture.GetWidth(), swap_chain_texture.GetHeight()}}, 1u, &clear_value};
    vkCmdBeginRenderPass(g_vulkan_context->GetCurrentCommandBuffer(), &rp, VK_SUBPASS_CONTENTS_INLINE);

    const VkViewport vp{
        0.0f, 0.0f, static_cast<float>(swap_chain_texture.GetWidth()), static_cast<float>(swap_chain_texture.GetHeight()), 0.0f, 1.0f};
    const VkRect2D scissor{{0, 0}, {static_cast<u32>(swap_chain_texture.GetWidth()), static_cast<u32>(swap_chain_texture.GetHeight())}};
    vkCmdSetViewport(g_vulkan_context->GetCurrentCommandBuffer(), 0, 1, &vp);
    vkCmdSetScissor(g_vulkan_context->GetCurrentCommandBuffer(), 0, 1, &scissor);
    return true;
}

void RendererVulkan::EndPresent() {
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData());

    VkCommandBuffer cmdbuffer = g_vulkan_context->GetCurrentCommandBuffer();
    vkCmdEndRenderPass(g_vulkan_context->GetCurrentCommandBuffer());
    m_swap_chain->GetCurrentTexture().TransitionToLayout(cmdbuffer, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    g_vulkan_context->SubmitCommandBuffer(m_swap_chain->GetImageAvailableSemaphore(), m_swap_chain->GetRenderingFinishedSemaphore(),
        m_swap_chain->GetSwapChain(), m_swap_chain->GetCurrentImageIndex(), !m_swap_chain->IsPresentModeSynchronizing());
    g_vulkan_context->MoveToNextCommandBuffer();
}

/// Initialize the renderer
VideoCore::ResultStatus RendererVulkan::Init() {
    // Create vulkan instance
    vk::ApplicationInfo app_info("PS2 Emulator", 1, nullptr, 0, VK_API_VERSION_1_3);

    // Get required extensions
    auto extensions = RequiredExtensions(render_window.GetWindowInfo().type, true);

    const char* layers = "VK_LAYER_KHRONOS_validation";
    vk::InstanceCreateInfo instance_info{{}, &app_info, layers, extensions};

    auto instance = vk::createInstance(instance_info);
    auto surface = swapchain->GetSurface();
    auto physical_device = instance.enumeratePhysicalDevices()[0];

    // Create global instance
    g_vk_instace = std::make_unique<VKInstance>();
    g_vk_instace->Create(instance, physical_device, surface, true);

    // Create Vulkan state and task manager
    VulkanState::Create();
    g_vk_task_scheduler = std::make_unique<VKTaskScheduler>();
    g_vk_task_scheduler->Create();

    auto& telemetry_session = Core::System::GetInstance().TelemetrySession();
    constexpr auto user_system = Common::Telemetry::FieldType::UserSystem;
    telemetry_session.AddField(user_system, "GPU_Vendor", "NVIDIA");
    telemetry_session.AddField(user_system, "GPU_Model", "GTX 1650");
    telemetry_session.AddField(user_system, "GPU_Vulkan_Version", "Vulkan 1.3");

    // Initialize the renderer
    CreateVulkanObjects();
    RefreshRasterizerSetting();

    return VideoCore::ResultStatus::Success;
}

/// Shutdown the renderer
void RendererVulkan::ShutDown() {}

} // namespace OpenGL
