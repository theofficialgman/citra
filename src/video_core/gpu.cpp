// Copyright 2015 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cstring>
#include <type_traits>
#include "core/core.h"
#include "video_core/pica.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_opengl/renderer_opengl.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"

std::unique_ptr<VideoCore::RendererBase> CreateRenderer(Core::System& system,
                                                        Frontend::EmuWindow& emu_window) {
    auto& telemetry_session = system.TelemetrySession();
    auto& cpu_memory = system.Memory();

    switch (Settings::values.renderer_backend) {
    case Settings::RendererBackend::OpenGL:
        return std::make_unique<OpenGL::RendererOpenGL>(emu_window);
    case Settings::RendererBackend::Vulkan:
        return std::make_unique<Vulkan::RendererVulkan>(emu_window);
    default:
        return nullptr;
    }
}

namespace Pica {

GPU::GPU(Core::System& system, Memory::MemorySystem& memory) :
    system(system), memory(memory) {
    //renderer = CreateRenderer(system, )
    rasterizer = renderer->Rasterizer();
}

void GPU::SwapBuffers() {
    renderer->SwapBuffers();
}

void GPU::FlushAll() {
    rasterizer->FlushAll();
}

void GPU::FlushRegion(PAddr addr, u32 size) {
    rasterizer->FlushRegion(addr, size);
}

void GPU::InvalidateRegion(PAddr addr, u32 size) {
    rasterizer->InvalidateRegion(addr, size);
}

void GPU::FlushAndInvalidateRegion(PAddr addr, u32 size) {
    rasterizer->FlushAndInvalidateRegion(addr, size);
}

void GPU::ClearAll(bool flush) {
    rasterizer->ClearAll(flush);
}

} // namespace Pica
