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
#include <queue>
#include <glm/gtc/matrix_transform.hpp>
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
        /*const VkWaylandSurfaceCreateInfoKHR wayland_ci{
            VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR, nullptr, 0,
            static_cast<wl_display*>(window_info.display_connection),
            static_cast<wl_surface*>(window_info.render_surface)};
        if (vkCreateWaylandSurfaceKHR(instance, &wayland_ci, nullptr, &unsafe_surface) != VK_SUCCESS) {
            LOG_ERROR(Render_Vulkan, "Failed to initialize Wayland surface");
            UNREACHABLE();
        }*/
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

/**
 * Defines a 1:1 pixel ortographic projection matrix with (0,0) on the top-left
 * corner and (width, height) on the lower-bottom.
 *
 * The projection part of the matrix is trivial, hence these operations are represented
 * by a 3x2 matrix.
 *
 * @param flipped Whether the frame should be flipped upside down.
 */
static glm::mat3x2 MakeOrthographicMatrix(const float width, const float height, bool flipped) {
    glm::mat3x2 matrix; // Laid out in column-major order

    // Last matrix row is implicitly assumed to be [0, 0, 1].
    if (flipped) {
        // clang-format off
        matrix[0][0] = 2.f / width; matrix[1][0] = 0.f;           matrix[2][0] = -1.f;
        matrix[0][1] = 0.f;         matrix[1][1] = 2.f / height;  matrix[2][1] = -1.f;
        // clang-format on
    }
    else {
        // clang-format off
        matrix[0][0] = 2.f / width; matrix[1][0] = 0.f;           matrix[2][0] = -1.f;
        matrix[1][1] = 0.f;         matrix[1][1] = -2.f / height; matrix[2][1] = 1.f;
        // clang-format on
    }

    return matrix;
}

RendererVulkan::RendererVulkan(Frontend::EmuWindow& window)
    : RendererBase{window} {

    window.mailbox = nullptr;
}

void RendererVulkan::PrepareRendertarget() {
    for (int i = 0; i < 3; i++) {
        int fb_id = i == 2 ? 1 : 0;
        const auto& framebuffer = GPU::g_regs.framebuffer_config[fb_id];

        // Create swapchain if needed
        if (swapchain->NeedsRecreation()) {
            swapchain->Create(framebuffer.width, framebuffer.height, false);
        }

        // Main LCD (0): 0x1ED02204, Sub LCD (1): 0x1ED02A04
        u32 lcd_color_addr =
            (fb_id == 0) ? LCD_REG_INDEX(color_fill_top) : LCD_REG_INDEX(color_fill_bottom);
        lcd_color_addr = HW::VADDR_LCD + 4 * lcd_color_addr;
        LCD::Regs::ColorFill color_fill = {0};
        LCD::Read(color_fill.raw, lcd_color_addr);

        if (color_fill.is_enabled) {
            LoadColorToActiveGLTexture(color_fill.color_r, color_fill.color_g, color_fill.color_b, screen_infos[i]);
        } else {
            auto [width, height] = screen_infos[i].texture.GetArea().extent;
            u32 fwidth = framebuffer.width;
            u32 fheight = framebuffer.height;

            if (width != fwidth || height != fheight ||
                    screen_infos[i].format != framebuffer.color_format) {
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
        ASSERT(false);
        // Reset the screen info's display texture to its own permanent texture
        screen_info.display_texture = &screen_info.texture;
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
    clear_color = vk::ClearColorValue{std::array<float, 4>{Settings::values.bg_red, Settings::values.bg_green, Settings::values.bg_blue, 0.0f}};
    clear_color = vk::ClearColorValue(std::array<float, 4>{1.0f, 0.0f, 0.0, 1.0f});

    //filter_sampler.Create();
    //ReloadSampler();

    // Generate VBO handle for drawing
    VKBuffer::Info vertex_info{
        .size = sizeof(ScreenRectVertex) * 10,
        .properties = vk::MemoryPropertyFlagBits::eDeviceLocal,
        .usage = vk::BufferUsageFlagBits::eVertexBuffer |
                 vk::BufferUsageFlagBits::eTransferDst
    };
    vertex_buffer.Create(vertex_info);
}

void RendererVulkan::ConfigureFramebufferTexture(ScreenInfo& screen, const GPU::Regs::FramebufferConfig& framebuffer) {
    screen.format = framebuffer.color_format;

    VKTexture::Info texture_info{
        .width = framebuffer.width,
        .height = framebuffer.height,
        .type = vk::ImageType::e2D,
        .view_type = vk::ImageViewType::e2D,
        .usage = vk::ImageUsageFlagBits::eColorAttachment |
                 vk::ImageUsageFlagBits::eTransferDst |
                 vk::ImageUsageFlagBits::eSampled
    };

    switch (screen.format) {
    case GPU::Regs::PixelFormat::RGBA8:
        texture_info.format = vk::Format::eR8G8B8A8Srgb;
        break;

    case GPU::Regs::PixelFormat::RGB8:
        // Note that the texture will actually give us an RGBA8 image because next to no modern hardware supports RGB formats.
        // The pixels will be converted automatically by Upload()
        texture_info.format = vk::Format::eR8G8B8Srgb;
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

    auto cmdbuffer = g_vk_task_scheduler->GetUploadCommandBuffer();
    texture.Transition(cmdbuffer, vk::ImageLayout::eShaderReadOnlyOptimal);
}

/**
 * Draws a single texture to the emulator window, rotating the texture to correct for the 3DS's LCD
 * rotation.
 */
void RendererVulkan::DrawSingleScreenRotated(u32 screen_id, float x, float y, float w, float h) {
    auto& screen_info = screen_infos[screen_id];
    const auto& texcoords = screen_info.display_texcoords;

    u32 size = sizeof(ScreenRectVertex) * 4;
    auto [ptr, offset, invalidate] = vertex_buffer.Map(size);

    const std::array vertices{
        ScreenRectVertex(x, y, texcoords.bottom, texcoords.left, screen_id),
        ScreenRectVertex(x + w, y, texcoords.bottom, texcoords.right, screen_id),
        ScreenRectVertex(x, y + h, texcoords.top, texcoords.left, screen_id),
        ScreenRectVertex(x + w, y + h, texcoords.top, texcoords.right, screen_id),
    };

    std::memcpy(ptr, vertices.data(), size);
    vertex_buffer.Commit(size, vk::AccessFlagBits::eVertexAttributeRead,
                         vk::PipelineStageFlagBits::eVertexInput);

    // As this is the "DrawSingleScreenRotated" function, the output resolution dimensions have been
    // swapped. If a non-rotated draw-screen function were to be added for book-mode games, those
    // should probably be set to the standard (w, h, 1.0 / w, 1.0 / h) ordering.
    const u16 scale_factor = VideoCore::GetResolutionScaleFactor();
    auto [width, height] = screen_info.texture.GetArea().extent;

    draw_info.i_resolution = glm::vec4{width * scale_factor, height * scale_factor,
                                       1.0f / (width * scale_factor),
                                       1.0f / (height * scale_factor)};
    draw_info.o_resolution = glm::vec4{h, w, 1.0f / h, 1.0f / w};

    auto& state = VulkanState::Get();
    state.SetPresentData(draw_info);

    auto cmdbuffer = g_vk_task_scheduler->GetRenderCommandBuffer();

    cmdbuffer.bindVertexBuffers(0, vertex_buffer.GetBuffer(), {0});
    cmdbuffer.draw(4, 1, offset / sizeof(ScreenRectVertex), 0);
}

void RendererVulkan::DrawSingleScreen(u32 screen_id, float x, float y, float w, float h) {
    auto& screen_info = screen_infos[screen_id];
    const auto& texcoords = screen_info.display_texcoords;

    u32 size = sizeof(ScreenRectVertex) * 4;
    auto [ptr, offset, invalidate] = vertex_buffer.Map(size);

    const std::array vertices{
        ScreenRectVertex(x, y, texcoords.bottom, texcoords.right, screen_id),
        ScreenRectVertex(x + w, y, texcoords.top, texcoords.right, screen_id),
        ScreenRectVertex(x, y + h, texcoords.bottom, texcoords.left, screen_id),
        ScreenRectVertex(x + w, y + h, texcoords.top, texcoords.left, screen_id),
    };

    std::memcpy(ptr, vertices.data(), size);
    vertex_buffer.Commit(size, vk::AccessFlagBits::eVertexAttributeRead,
                         vk::PipelineStageFlagBits::eVertexInput);

    const u16 scale_factor = VideoCore::GetResolutionScaleFactor();
    auto [width, height] = screen_info.texture.GetArea().extent;


    draw_info.i_resolution = glm::vec4{width * scale_factor, height * scale_factor,
                                       1.0f / (width * scale_factor),
                                       1.0f / (height * scale_factor)};
    draw_info.o_resolution = glm::vec4{h, w, 1.0f / h, 1.0f / w};

    auto& state = VulkanState::Get();
    state.SetPresentData(draw_info);

    auto cmdbuffer = g_vk_task_scheduler->GetRenderCommandBuffer();

    cmdbuffer.bindVertexBuffers(0, vertex_buffer.GetBuffer(), {0});
    cmdbuffer.draw(4, 1, offset / sizeof(ScreenRectVertex), 0);
}

/**
 * Draws a single texture to the emulator window, rotating the texture to correct for the 3DS's LCD
 * rotation.
 */
void RendererVulkan::DrawSingleScreenStereoRotated(const ScreenInfo& screen_info_l,
                                                   const ScreenInfo& screen_info_r, float x,
                                                   float y, float w, float h) {
    ASSERT(false);
    //DrawSingleScreenRotated(screen_info_l, x, y, w, h);
    /*const auto& texcoords = screen_info_l.display_texcoords;

    const std::array<ScreenRectVertex, 4> vertices = {{
        ScreenRectVertex(x, y, texcoords.bottom, texcoords.left),
        ScreenRectVertex(x + w, y, texcoords.bottom, texcoords.right),
        ScreenRectVertex(x, y + h, texcoords.top, texcoords.left),
        ScreenRectVertex(x + w, y + h, texcoords.top, texcoords.right),
    }};

    const u16 scale_factor = VideoCore::GetResolutionScaleFactor();
    auto [width_l, height_l] = screen_info_l.texture.GetArea().extent;

    draw_info.i_resolution = glm::vec4{width_l * scale_factor, height_l * scale_factor,
                                       1.0f / (width_l * scale_factor),
                                       1.0f / (height_l * scale_factor)};
    draw_info.o_resolution = glm::vec4{h, w, 1.0f / h, 1.0f / w};

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
    state.Apply();*/
}

void RendererVulkan::DrawSingleScreenStereo(const ScreenInfo& screen_info_l,
                                            const ScreenInfo& screen_info_r, float x, float y,
                                            float w, float h) {
    ASSERT(false);
    //DrawSingleScreen(screen_info_l, x, y, w, h);
    /*const auto& texcoords = screen_info_l.display_texcoords;

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
    state.Apply();*/
}

/**
 * Draws the emulated screens to the emulator window.
 */
void RendererVulkan::DrawScreens(const Layout::FramebufferLayout& layout, bool flipped) {
    if (VideoCore::g_renderer_bg_color_update_requested.exchange(false)) {
        // Update background color before drawing
        clear_color = vk::ClearColorValue{std::array<float, 4>{Settings::values.bg_red, Settings::values.bg_green, Settings::values.bg_blue, 0.0f}};
    }

    if (VideoCore::g_renderer_sampler_update_requested.exchange(false)) {
        // Set the new filtering mode for the sampler
        //ReloadSampler();
    }

    if (VideoCore::g_renderer_shader_update_requested.exchange(false)) {
        // Update fragment shader before drawing
        //shader.Release();
        // Link shaders and get variable locations
        //ReloadShader();
    }

    const auto& top_screen = layout.top_screen;
    const auto& bottom_screen = layout.bottom_screen;

    // Set projection matrix
    draw_info.modelview = glm::transpose(glm::ortho(0.f, static_cast<float>(layout.width),
                                                    static_cast<float>(layout.height), 0.0f,
                                                    0.f, 1.f));
    const bool stereo_single_screen = false
    /*    Settings::values.render_3d == Settings::StereoRenderOption::Anaglyph ||
        Settings::values.render_3d == Settings::StereoRenderOption::Interlaced ||
        Settings::values.render_3d == Settings::StereoRenderOption::ReverseInterlaced*/;

    // Bind a second texture for the right eye if in Anaglyph mode
    if (stereo_single_screen) {
        //glUniform1i(uniform_color_texture_r, 1);
    }

    auto& image = swapchain->GetCurrentImage();
    auto& state = VulkanState::Get();

    state.BeginRendering(image, std::nullopt, false, clear_color, vk::AttachmentLoadOp::eClear);
    state.SetPresentTextures(screen_infos[0].display_texture->GetView(),
                             screen_infos[1].display_texture->GetView(),
                             screen_infos[2].display_texture->GetView());
    state.ApplyPresentState();

    draw_info.layer = 0;
    if (layout.top_screen_enabled) {
        if (layout.is_rotated) {
            if (Settings::values.render_3d == Settings::StereoRenderOption::Off) {
                DrawSingleScreenRotated(0, top_screen.left,
                                        top_screen.top, top_screen.GetWidth(),
                                        top_screen.GetHeight());
            } else if (Settings::values.render_3d == Settings::StereoRenderOption::SideBySide) {
                DrawSingleScreenRotated(0, (float)top_screen.left / 2,
                                        (float)top_screen.top, (float)top_screen.GetWidth() / 2,
                                        (float)top_screen.GetHeight());
                draw_info.layer = 1;
                DrawSingleScreenRotated(1,
                                        ((float)top_screen.left / 2) + ((float)layout.width / 2),
                                        (float)top_screen.top, (float)top_screen.GetWidth() / 2,
                                        (float)top_screen.GetHeight());
            } else if (Settings::values.render_3d == Settings::StereoRenderOption::CardboardVR) {
                DrawSingleScreenRotated(0, layout.top_screen.left,
                                        layout.top_screen.top, layout.top_screen.GetWidth(),
                                        layout.top_screen.GetHeight());
                draw_info.layer = 1;
                DrawSingleScreenRotated(1,
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
                DrawSingleScreen(0, (float)top_screen.left, (float)top_screen.top,
                                 (float)top_screen.GetWidth(), (float)top_screen.GetHeight());
            } else if (Settings::values.render_3d == Settings::StereoRenderOption::SideBySide) {
                DrawSingleScreen(0, (float)top_screen.left / 2, (float)top_screen.top,
                                 (float)top_screen.GetWidth() / 2, (float)top_screen.GetHeight());
                draw_info.layer = 1;
                DrawSingleScreen(1,
                                 ((float)top_screen.left / 2) + ((float)layout.width / 2),
                                 (float)top_screen.top, (float)top_screen.GetWidth() / 2,
                                 (float)top_screen.GetHeight());
            } else if (Settings::values.render_3d == Settings::StereoRenderOption::CardboardVR) {
                DrawSingleScreen(0, layout.top_screen.left, layout.top_screen.top,
                                 layout.top_screen.GetWidth(), layout.top_screen.GetHeight());
                draw_info.layer = 1;
                DrawSingleScreen(1,
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

    draw_info.layer = 0;
    if (layout.bottom_screen_enabled) {
        if (layout.is_rotated) {
            if (Settings::values.render_3d == Settings::StereoRenderOption::Off) {
                DrawSingleScreenRotated(2, (float)bottom_screen.left,
                                        (float)bottom_screen.top, (float)bottom_screen.GetWidth(),
                                        (float)bottom_screen.GetHeight());
            } else if (Settings::values.render_3d == Settings::StereoRenderOption::SideBySide) {
                DrawSingleScreenRotated(
                    2, (float)bottom_screen.left / 2, (float)bottom_screen.top,
                    (float)bottom_screen.GetWidth() / 2, (float)bottom_screen.GetHeight());
                draw_info.layer = 1;
                DrawSingleScreenRotated(
                    2, ((float)bottom_screen.left / 2) + ((float)layout.width / 2),
                    (float)bottom_screen.top, (float)bottom_screen.GetWidth() / 2,
                    (float)bottom_screen.GetHeight());
            } else if (Settings::values.render_3d == Settings::StereoRenderOption::CardboardVR) {
                DrawSingleScreenRotated(2, layout.bottom_screen.left,
                                        layout.bottom_screen.top, layout.bottom_screen.GetWidth(),
                                        layout.bottom_screen.GetHeight());
                draw_info.layer = 1;
                DrawSingleScreenRotated(2,
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
                DrawSingleScreen(2, (float)bottom_screen.left,
                                 (float)bottom_screen.top, (float)bottom_screen.GetWidth(),
                                 (float)bottom_screen.GetHeight());
            } else if (Settings::values.render_3d == Settings::StereoRenderOption::SideBySide) {
                DrawSingleScreen(2, (float)bottom_screen.left / 2,
                                 (float)bottom_screen.top, (float)bottom_screen.GetWidth() / 2,
                                 (float)bottom_screen.GetHeight());
                draw_info.layer = 1;
                DrawSingleScreen(2,
                                 ((float)bottom_screen.left / 2) + ((float)layout.width / 2),
                                 (float)bottom_screen.top, (float)bottom_screen.GetWidth() / 2,
                                 (float)bottom_screen.GetHeight());
            } else if (Settings::values.render_3d == Settings::StereoRenderOption::CardboardVR) {
                DrawSingleScreen(2, layout.bottom_screen.left,
                                 layout.bottom_screen.top, layout.bottom_screen.GetWidth(),
                                 layout.bottom_screen.GetHeight());
                draw_info.layer = 1;
                DrawSingleScreen(2,
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

void RendererVulkan::SwapBuffers() {
    // Configure current framebuffer and recreate swapchain if necessary
    PrepareRendertarget();

    if (BeginPresent()) {
        const auto& layout = render_window.GetFramebufferLayout();
        DrawScreens(layout, false);

        EndPresent();
    }
}

bool RendererVulkan::BeginPresent() {
    swapchain->AcquireNextImage();

    auto& image = swapchain->GetCurrentImage();
    auto cmdbuffer = g_vk_task_scheduler->GetRenderCommandBuffer();

    // Swap chain images start in undefined
    image.OverrideImageLayout(vk::ImageLayout::eUndefined);
    image.Transition(cmdbuffer, vk::ImageLayout::eColorAttachmentOptimal);

    // Update viewport and scissor
    const auto [width, height] = image.GetArea().extent;
    const vk::Viewport viewport{0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f};
    const vk::Rect2D scissor{{0, 0}, {width, height}};

    cmdbuffer.setViewport(0, viewport);
    cmdbuffer.setScissor(0, scissor);

    return true;
}

void RendererVulkan::EndPresent() {
    auto& state = VulkanState::Get();
    state.EndRendering();

    auto& image = swapchain->GetCurrentImage();
    auto cmdbuffer = g_vk_task_scheduler->GetRenderCommandBuffer();
    image.Transition(cmdbuffer, vk::ImageLayout::ePresentSrcKHR);

    g_vk_task_scheduler->Submit(false, true, swapchain.get());
}

/// Initialize the renderer
VideoCore::ResultStatus RendererVulkan::Init() {
    // Create vulkan instance
    vk::ApplicationInfo app_info{"Citra", VK_MAKE_VERSION(1, 0, 0), nullptr, 0, VK_API_VERSION_1_3};

    // Get required extensions
    auto extensions = RequiredExtensions(render_window.GetWindowInfo().type, true);

    const char* layers = "VK_LAYER_KHRONOS_validation";
    vk::InstanceCreateInfo instance_info{{}, &app_info, layers, extensions};

    auto instance = vk::createInstance(instance_info);
    auto physical_devices = instance.enumeratePhysicalDevices();

    auto props = physical_devices[1].getProperties();
    std::cout << props.deviceName << '\n';

    // Create global instance
    auto surface = CreateSurface(instance, render_window);
    g_vk_instace = std::make_unique<VKInstance>();
    g_vk_task_scheduler = std::make_unique<VKTaskScheduler>();
    g_vk_instace->Create(instance, physical_devices[1], surface, true);
    g_vk_task_scheduler->Create();

    //auto& layout = render_window.GetFramebufferLayout();
    swapchain = std::make_shared<VKSwapChain>(surface);
    //swapchain->Create(layout.width, layout.height, false);

    // Create Vulkan state
    VulkanState::Create(swapchain);
    g_vk_task_scheduler->BeginTask();

    auto& telemetry_session = Core::System::GetInstance().TelemetrySession();
    constexpr auto user_system = Common::Telemetry::FieldType::UserSystem;
    telemetry_session.AddField(user_system, "GPU_Vendor", "NVIDIA");
    telemetry_session.AddField(user_system, "GPU_Model", "GTX 1650");
    telemetry_session.AddField(user_system, "GPU_Vulkan_Version", "Vulkan 1.3");

    // Initialize the renderer
    CreateVulkanObjects();
    RefreshRasterizerSetting();

    return static_cast<VideoCore::ResultStatus>(0);
}

/// Shutdown the renderer
void RendererVulkan::ShutDown() {}

} // namespace OpenGL
