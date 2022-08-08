// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/object_pool.h"
#include "common/vector_math.h"
#include "video_core/common/pipeline.h"
#include "video_core/common/framebuffer.h"

namespace Frontend {
class EmuWindow;
}

namespace VideoCore {

/// Common interface of a video backend
class BackendBase {
public:
    BackendBase(Frontend::EmuWindow& window) : window(window) {}
    virtual ~BackendBase() = default;

    // Triggers a swapchain buffer swap
    virtual void SwapBuffers();

    // Creates a backend specific texture handle
    virtual TextureHandle CreateTexture(TextureInfo info) = 0;

    // Creates a backend specific buffer handle
    virtual BufferHandle CreateBuffer(BufferInfo info) = 0;

    // Creates a backend specific framebuffer handle
    virtual FramebufferHandle CreateFramebuffer(FramebufferInfo info) = 0;

    // Creates a backend specific pipeline handle
    virtual PipelineHandle CreatePipeline(PipelineType type, PipelineInfo info) = 0;

    // Creates a backend specific sampler object
    virtual SamplerHandle CreateSampler(SamplerInfo info) = 0;

    // Start a draw operation
    virtual void Draw(PipelineHandle pipeline, FramebufferHandle draw_framebuffer,
                      BufferHandle vertex_buffer,
                      u32 base_vertex, u32 num_vertices) = 0;

    // Start an indexed draw operation
    virtual void DrawIndexed(PipelineHandle pipeline, FramebufferHandle draw_framebuffer,
                             BufferHandle vertex_buffer, BufferHandle index_buffer, AttribType index_type,
                             u32 base_index, u32 num_indices, u32 base_vertex) = 0;

    // Executes a compute shader
    virtual void DispatchCompute(PipelineHandle pipeline, Common::Vec3<u32> groupsize,
                                 Common::Vec3<u32> groups) = 0;

private:
    Frontend::EmuWindow& window;
};

} // namespace VideoCore
