// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/vector_math.h"
#include "video_core/common/pipeline.h"
#include "video_core/common/framebuffer.h"

namespace Frontend {
class EmuWindow;
}

namespace VideoCore {

// A piece of information the video frontend can query the backend about
enum class Query {
    PresentFormat = 0
};

// Common interface of a video backend
class BackendBase {
public:
    BackendBase(Frontend::EmuWindow& window) : window(window) {}
    virtual ~BackendBase() = default;

    // Acquires the next swapchain images and begins rendering
    virtual bool BeginPresent() = 0;

    // Triggers a swapchain buffer swap
    virtual void EndPresent() = 0;

    // Returns the framebuffer created from the swapchain images
    virtual FramebufferHandle GetWindowFramebuffer() = 0;

    // Asks the driver about a particular piece of information
    virtual u64 QueryDriver(Query query) = 0;

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

    // Creates a backend specific shader object
    virtual ShaderHandle CreateShader(ShaderStage stage, std::string_view name, std::string source) = 0;

    // Binds a vertex buffer at a provided offset
    virtual void BindVertexBuffer(BufferHandle buffer, std::span<const u32> offsets) = 0;

    // Binds an index buffer at provided offset
    virtual void BindIndexBuffer(BufferHandle buffer, AttribType index_type, u32 offset) = 0;

    // Start a draw operation
    virtual void Draw(PipelineHandle pipeline, FramebufferHandle draw_framebuffer, u32 base_vertex,
                      u32 num_vertices) = 0;

    // Start an indexed draw operation
    virtual void DrawIndexed(PipelineHandle pipeline, FramebufferHandle draw_framebuffer, u32 base_vertex,
                             u32 base_index, u32 num_indices) = 0;

    // Executes a compute shader
    virtual void DispatchCompute(PipelineHandle pipeline, Common::Vec3<u32> groupsize,
                                 Common::Vec3<u32> groups) = 0;
protected:
    Frontend::EmuWindow& window;
};

} // namespace VideoCore
