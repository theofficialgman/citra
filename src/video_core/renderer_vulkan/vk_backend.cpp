// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#define VULKAN_HPP_NO_CONSTRUCTORS
#include "core/core.h"
#include "common/object_pool.h"
#include "video_core/renderer_vulkan/vk_backend.h"
#include "video_core/renderer_vulkan/vk_buffer.h"
#include "video_core/renderer_vulkan/vk_texture.h"
#include "video_core/renderer_vulkan/vk_framebuffer.h"

namespace VideoCore::Vulkan {

constexpr vk::PipelineBindPoint ToVkPipelineBindPoint(PipelineType type) {
    switch (type) {
    case PipelineType::Graphics:
        return vk::PipelineBindPoint::eGraphics;
    case PipelineType::Compute:
        return vk::PipelineBindPoint::eCompute;
    }
}

constexpr vk::Rect2D ToVkRect2D(Rect2D rect) {
    return vk::Rect2D{
        .offset = vk::Offset2D{rect.x, rect.y},
        .extent = vk::Extent2D{rect.width, rect.height}
    };
}

constexpr vk::IndexType ToVkIndexType(AttribType type) {
    switch (type) {
    case AttribType::Short:
        return vk::IndexType::eUint16;
    case AttribType::Int:
        return vk::IndexType::eUint32;
    default:
        LOG_CRITICAL(Render_Vulkan, "Unknown index type {}!", type);
        UNREACHABLE();
    }
}

Backend::Backend(Frontend::EmuWindow& window) : BackendBase(window),
    instance(window), swapchain(instance, instance.GetSurface()),
    scheduler(instance), renderpass_cache(instance, swapchain) {

    // TODO: Properly report GPU hardware
    auto& telemetry_session = Core::System::GetInstance().TelemetrySession();
    constexpr auto user_system = Common::Telemetry::FieldType::UserSystem;
    telemetry_session.AddField(user_system, "GPU_Vendor", "NVIDIA");
    telemetry_session.AddField(user_system, "GPU_Model", "GTX 1650");
    telemetry_session.AddField(user_system, "GPU_Vulkan_Version", "Vulkan 1.3");

    // Create pipeline cache object
    vk::Device device = instance.GetDevice();
    pipeline_cache = device.createPipelineCache({});

    constexpr std::array pool_sizes = {
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 1024},
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBufferDynamic, 1024},
        vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 2048},
        vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 2048},
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformTexelBuffer, 1024}
    };

    const vk::DescriptorPoolCreateInfo pool_info = {
        .maxSets = 2048,
        .poolSizeCount = pool_sizes.size(),
        .pPoolSizes = pool_sizes.data()
    };

    // Create descriptor pools
    for (u32 pool = 0; pool < SCHEDULER_COMMAND_COUNT; pool++) {
        descriptor_pools[pool] = device.createDescriptorPool(pool_info);
    }
}

Backend::~Backend() {
    vk::Device device = instance.GetDevice();
    device.destroyPipelineCache(pipeline_cache);

    for (u32 pool = 0; pool < SCHEDULER_COMMAND_COUNT; pool++) {
        device.destroyDescriptorPool(descriptor_pools[pool]);
    }
}

u64 Backend::PipelineInfoHash(const PipelineInfo& info) {
    const bool hash_all = !instance.IsExtendedDynamicStateSupported();
    if (hash_all) {
        // Don't hash the last three members of DepthStencilState, these are
        // dynamic in every Vulkan implementation
        return Common::ComputeHash64(&info, offsetof(PipelineInfo, depth_stencil) +
                                            offsetof(DepthStencilState, stencil_reference));
    } else {
        // Hash everything except depth_stencil and rasterization
        return Common::ComputeHash64(&info, offsetof(PipelineInfo, rasterization));
    }
}

/**
 * To avoid many small heap allocations during handle creation, each resource has a dedicated pool
 * associated with it that batch-allocates memory.
 */
BufferHandle Backend::CreateBuffer(BufferInfo info) {
    static ObjectPool<Buffer> buffer_pool;
    return BufferHandle{buffer_pool.Allocate(instance, scheduler, info)};
}

FramebufferHandle Backend::CreateFramebuffer(FramebufferInfo info) {
    static ObjectPool<Framebuffer> framebuffer_pool;

    // Get renderpass
    TextureFormat color = info.color.IsValid() ? info.color->GetFormat() : TextureFormat::Undefined;
    TextureFormat depth = info.depth_stencil.IsValid() ? info.depth_stencil->GetFormat() : TextureFormat::Undefined;
    vk::RenderPass load_renderpass = GetRenderPass(color, depth, false);
    vk::RenderPass clear_renderpass = GetRenderPass(color, depth, true);

    return FramebufferHandle{framebuffer_pool.Allocate(instance, info, load_renderpass, clear_renderpass)};
}

TextureHandle Backend::CreateTexture(TextureInfo info) {
    static ObjectPool<Texture> texture_pool;
    return TextureHandle{texture_pool.Allocate(instance, scheduler, info)};
}

PipelineHandle Backend::CreatePipeline(PipelineType type, PipelineInfo info) {
    static ObjectPool<Pipeline> pipeline_pool;

    // Get renderpass
    vk::RenderPass renderpass = GetRenderPass(info.color_attachment, info.depth_attachment);

    // Find an owner first
    if (auto iter = pipeline_owners.find(info.layout); iter != pipeline_owners.end()) {
        return PipelineHandle{pipeline_pool.Allocate(instance, iter->second, type, info,
                                                     renderpass, pipeline_cache)};
    }

    // Create the layout
    auto result = pipeline_owners.emplace(info.layout, PipelineOwner{instance, info.layout});
    return PipelineHandle{pipeline_pool.Allocate(instance, result.first->second, type, info,
                                                 renderpass, pipeline_cache)};
}

SamplerHandle Backend::CreateSampler(SamplerInfo info) {
    static ObjectPool<Sampler> sampler_pool;
    return SamplerHandle{sampler_pool.Allocate(info)};
}

void Backend::BindVertexBuffer(BufferHandle buffer, std::span<const u64> offsets) {
    const Buffer* vertex = static_cast<const Buffer*>(buffer.Get());

    std::array<vk::Buffer, 16> buffers;
    buffers.fill(vertex->GetHandle());

    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.bindVertexBuffers(0, offsets.size(), buffers.data(), offsets.data());
}

void Backend::BindIndexBuffer(BufferHandle buffer, AttribType index_type, u64 offset) {
    const Buffer* index = static_cast<const Buffer*>(buffer.Get());

    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.bindIndexBuffer(index->GetHandle(), 0, ToVkIndexType(index_type));
}

void Backend::Draw(PipelineHandle pipeline_handle, FramebufferHandle draw_framebuffer,
                   u32 base_vertex, u32 num_vertices) {

    // Bind descriptor sets
    BindDescriptorSets(pipeline_handle);

    // Begin renderpass
    BeginRenderpass(draw_framebuffer);

    // Bind pipeline
    const Pipeline* pipeline = static_cast<const Pipeline*>(pipeline_handle.Get());
    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.bindPipeline(ToVkPipelineBindPoint(pipeline->GetType()), pipeline->GetHandle());

    // Submit draw
    command_buffer.draw(num_vertices, 1, base_vertex, 0);

    // End renderpass
    command_buffer.endRenderPass();
}

void Backend::DrawIndexed(PipelineHandle pipeline_handle, FramebufferHandle draw_framebuffer,
                          u32 base_index, u32 num_indices, u32 base_vertex) {
    // Bind descriptor sets
    BindDescriptorSets(pipeline_handle);

    // Begin renderpass
    BeginRenderpass(draw_framebuffer);

    // Bind pipeline
    const Pipeline* pipeline = static_cast<const Pipeline*>(pipeline_handle.Get());
    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.bindPipeline(ToVkPipelineBindPoint(pipeline->GetType()), pipeline->GetHandle());

    // Submit draw
    command_buffer.drawIndexed(num_indices, 1, base_index, base_vertex, 0);

    // End renderpass
    command_buffer.endRenderPass();

}

vk::RenderPass Backend::GetRenderPass(TextureFormat color, TextureFormat depth, bool is_clear) const {
    if (color == TextureFormat::PresentColor) {
        return renderpass_cache.GetPresentRenderpass();
    } else {
        return renderpass_cache.GetRenderpass(color, depth, is_clear);
    }
}

void Backend::BindDescriptorSets(PipelineHandle handle) {
    Pipeline* pipeline = static_cast<Pipeline*>(handle.Get());
    PipelineOwner& pipeline_owner = pipeline->GetOwner();

    std::array<vk::DescriptorSet, MAX_BINDING_GROUPS> bound_sets;
    const u32 set_count = pipeline_owner.GetDescriptorSetLayoutCount();
    for (int i = 0; i < set_count; i++) {
        if (!pipeline_owner.descriptor_dirty[i]) {
            // Get the ready descriptor if it hasn't been modified
            bound_sets[i] = pipeline_owner.descriptor_bank[i];
        } else {
            // Otherwise allocate a new set and update it with the needed data
            u32 pool_index = scheduler.GetCurrentSlotIndex();
            const vk::DescriptorSetAllocateInfo alloc_info = {
                .descriptorPool = descriptor_pools[pool_index],
                .descriptorSetCount = 1,
                .pSetLayouts = &pipeline_owner.GetDescriptorSetLayouts()[i]
            };

            vk::Device device = instance.GetDevice();
            vk::DescriptorSet set = device.allocateDescriptorSets(alloc_info)[0];
            device.updateDescriptorSetWithTemplate(set, pipeline_owner.GetUpdateTemplate(i),
                                                   pipeline_owner.GetData(i));

            bound_sets[i] = set;
            pipeline_owner.descriptor_bank[i] = set;
            pipeline_owner.descriptor_dirty[i] = false;
        }
    }

    // Bind the descriptor sets
    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.bindDescriptorSets(ToVkPipelineBindPoint(handle->GetType()), pipeline_owner.GetLayout(),
                                      0, set_count, bound_sets.data(), 0, nullptr);
}

void Backend::BeginRenderpass(FramebufferHandle draw_framebuffer) {
    const Framebuffer* framebuffer = static_cast<const Framebuffer*>(draw_framebuffer.Get());

    u32 clear_value_count = 0;
    std::array<vk::ClearValue, 2> clear_values{};

    if (framebuffer->GetColorAttachment().IsValid()) {
        for (int i = 0; i < 4; i++) {
            clear_values[clear_value_count++].color.float32[i] = framebuffer->clear_color_value[i];
        }
    }

    if (framebuffer->GetDepthStencilAttachment().IsValid()) {
        clear_values[clear_value_count].depthStencil.depth = framebuffer->clear_depth_value;
        clear_values[clear_value_count++].depthStencil.stencil = framebuffer->clear_stencil_value;
    }

    // Use the clear renderpass if the framebuffer was configured so
    const vk::RenderPassBeginInfo renderpass_begin = {
        .renderPass = framebuffer->GetLoadOp() == LoadOp::Load ?
                      framebuffer->GetLoadRenderpass() :
                      framebuffer->GetClearRenderpass(),
        .framebuffer = framebuffer->GetHandle(),
        .renderArea = ToVkRect2D(framebuffer->GetDrawRect()),
        .clearValueCount = clear_value_count,
        .pClearValues = clear_values.data()
    };

    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.beginRenderPass(renderpass_begin, vk::SubpassContents::eInline);
}

} // namespace VideoCore::Vulkan
