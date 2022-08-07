// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <functional>
#include "core/frontend/framebuffer_layout.h"
#include "video_core/maestro.h"

namespace Core {
class System;
}

namespace Memory {
class MemorySystem;
}

namespace Frontend {
class EmuWindow;
}

namespace VideoCore {
class RendererBase;
class RasterizerInterface;
}

namespace Pica {

class Maestro;

enum class ResultStatus {
    Success,
    ErrorGenericDrivers,
    ErrorUnsupportedGL,
};

/**
 * Interface for the PICA GPU
 */
class GPU {
public:
    GPU(Core::System& system, Memory::MemorySystem& memory);
    ~GPU() = default;

    /// Swap buffers (render frame)
    void SwapBuffers();

    /// Notify rasterizer that all caches should be flushed to 3DS memory
    void FlushAll();

    /// Notify rasterizer that any caches of the specified region should be flushed to 3DS memory
    void FlushRegion(PAddr addr, u32 size);

    /// Notify rasterizer that any caches of the specified region should be invalidated
    void InvalidateRegion(PAddr addr, u32 size);

    /// Notify rasterizer that any caches of the specified region should be flushed and invalidated
    void FlushAndInvalidateRegion(PAddr addr, u32 size);

    /// Removes as much state as possible from the rasterizer in preparation for a save/load state
    void ClearAll(bool flush);

    /// Request a screenshot of the next frame
    void RequestScreenshot(u8* data, std::function<void()> callback,
                           const Layout::FramebufferLayout& layout);

    /// Returns the resolution scale factor
    u16 GetResolutionScaleFactor();

private:
    Core::System& system;
    Memory::MemorySystem& memory;

    // Renderer
    VideoCore::RasterizerInterface* rasterizer = nullptr;
    std::unique_ptr<VideoCore::RendererBase> renderer = nullptr;
    std::unique_ptr<Maestro> maestro = nullptr;
};

} // namespace VideoCore
