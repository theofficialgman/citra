// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#define VULKAN_HPP_NO_CONSTRUCTORS
#include <functional>
#include "common/file_util.h"
#include "common/linear_disk_cache.h"
#include "core/core.h"
#include "core/frontend/emu_window.h"
#include "video_core/renderer_vulkan/vk_backend.h"
#include "video_core/renderer_vulkan/vk_buffer.h"
#include "video_core/renderer_vulkan/vk_texture.h"
#include "video_core/renderer_vulkan/vk_framebuffer.h"
#include "video_core/renderer_vulkan/vk_shader.h"

namespace VideoCore::Vulkan {

constexpr vk::PipelineBindPoint ToVkPipelineBindPoint(PipelineType type) {
    switch (type) {
    case PipelineType::Graphics:
        return vk::PipelineBindPoint::eGraphics;
    case PipelineType::Compute:
        return vk::PipelineBindPoint::eCompute;
    }
}

inline vk::Rect2D ToVkRect2D(Rect2D rect) {
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

class PipelineCacheReadCallback : public LinearDiskCacheReader<u32, u8>
{
public:
  PipelineCacheReadCallback(std::vector<u8>* data) : m_data(data) {}
  void Read(const u32& key, const u8* value, u32 value_size) override
  {
    m_data->resize(value_size);
    if (value_size > 0)
      memcpy(m_data->data(), value, value_size);
  }

private:
  std::vector<u8>* m_data;
};

class PipelineCacheReadIgnoreCallback : public LinearDiskCacheReader<u32, u8>
{
public:
  void Read(const u32& key, const u8* value, u32 value_size) override {}
};


Backend::Backend(Frontend::EmuWindow& window) : BackendBase(window),
    instance(window), scheduler(instance, pool_manager), renderpass_cache(instance),
    swapchain(instance, scheduler, renderpass_cache, pool_manager, instance.GetSurface()) {

    // TODO: Properly report GPU hardware
    auto& telemetry_session = Core::System::GetInstance().TelemetrySession();
    constexpr auto user_system = Common::Telemetry::FieldType::UserSystem;
    telemetry_session.AddField(user_system, "GPU_Vendor", "NVIDIA");
    telemetry_session.AddField(user_system, "GPU_Model", "GTX 1650");
    telemetry_session.AddField(user_system, "GPU_Vulkan_Version", "Vulkan 1.3");

    vk::PipelineCacheCreateInfo cache_info{};

    std::vector<u8> disk_data;
    LinearDiskCache<u32, u8> disk_cache;
    PipelineCacheReadCallback read_callback(&disk_data);
    if (disk_cache.OpenAndRead(pipeline_cache_filename.c_str(), read_callback) != 1) {
        disk_data.clear();
    }

    if (!disk_data.empty()) {
        // Don't use this data. In fact, we should delete it to prevent it from being used next time.
        FileUtil::Delete(pipeline_cache_filename);
    } else {
        cache_info.initialDataSize = disk_data.size();
        cache_info.pInitialData = disk_data.data();
    }

    // Create pipeline cache object
    vk::Device device = instance.GetDevice();
    pipeline_cache = device.createPipelineCache(cache_info);

    constexpr std::array pool_sizes = {
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 1024},
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBufferDynamic, 1024},
        vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 2048},
        vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 2048},
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformTexelBuffer, 1024}
    };

    const vk::DescriptorPoolCreateInfo pool_info = {
        .maxSets = 2048,
        .poolSizeCount = static_cast<u32>(pool_sizes.size()),
        .pPoolSizes = pool_sizes.data()
    };

    // Create descriptor pools
    for (u32 pool = 0; pool < SCHEDULER_COMMAND_COUNT; pool++) {
        descriptor_pools[pool] = device.createDescriptorPool(pool_info);
    }

    auto callback = std::bind(&Backend::OnCommandSwitch, this, std::placeholders::_1);
    scheduler.SetSwitchCallback(callback);
}

Backend::~Backend() {
    // Wait for all GPU operations to finish before continuing
    vk::Device device = instance.GetDevice();
    device.waitIdle();

    auto data = device.getPipelineCacheData(pipeline_cache);

    // Delete the old cache and re-create.
    FileUtil::Delete(pipeline_cache_filename);

    // We write a single key of 1, with the entire pipeline cache data.
    // Not ideal, but our disk cache class does not support just writing a single blob
    // of data without specifying a key.
    LinearDiskCache<u32, u8> disk_cache;
    PipelineCacheReadIgnoreCallback callback;
    disk_cache.OpenAndRead(pipeline_cache_filename.c_str(), callback);
    disk_cache.Append(1, data.data(), static_cast<u32>(data.size()));
    disk_cache.Close();

    device.destroyPipelineCache(pipeline_cache);

    for (u32 pool = 0; pool < SCHEDULER_COMMAND_COUNT; pool++) {
        device.destroyDescriptorPool(descriptor_pools[pool]);
    }
}

bool Backend::BeginPresent() {
    const auto& layout = window.GetFramebufferLayout();
    if (swapchain.NeedsRecreation()) {
        swapchain.Create(layout.width, layout.height, false);
    }

    swapchain.AcquireNextImage();
    return true;
}

void Backend::EndPresent() {
    // Transition swapchain image to present layout
    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    /*if (renderpass_active) {
        command_buffer.endRenderPass();
        renderpass_active = false;
    }*/

    swapchain.GetCurrentImage()->Transition(command_buffer, vk::ImageLayout::ePresentSrcKHR);

    // Submit and present
    scheduler.Submit(false, true, swapchain.GetAvailableSemaphore(), swapchain.GetPresentSemaphore());
    swapchain.Present();
}

void Backend::Flush() {
    scheduler.Submit(true);
}

FramebufferHandle Backend::GetWindowFramebuffer() {
    auto framebuffer = swapchain.GetCurrentFramebuffer();
    auto extent = swapchain.GetExtent();
    framebuffer->SetDrawRect({0, extent.height, extent.width, 0});

    return framebuffer;
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
    return pool_manager.Allocate<Buffer>(instance, scheduler, pool_manager, info);
}

FramebufferHandle Backend::CreateFramebuffer(FramebufferInfo info) {
    // Get renderpass
    TextureFormat color = info.color.IsValid() ? info.color->GetFormat() : TextureFormat::Undefined;
    TextureFormat depth = info.depth_stencil.IsValid() ? info.depth_stencil->GetFormat() : TextureFormat::Undefined;
    vk::RenderPass load_renderpass = GetRenderPass(color, depth, false);
    vk::RenderPass clear_renderpass = GetRenderPass(color, depth, true);

    return pool_manager.Allocate<Framebuffer>(instance, scheduler, pool_manager, info,
                                              load_renderpass, clear_renderpass);
}

TextureHandle Backend::CreateTexture(TextureInfo info) {
    return pool_manager.Allocate<Texture>(instance, scheduler, pool_manager, info);
}

PipelineHandle Backend::CreatePipeline(PipelineType type, PipelineInfo info) {
    // Get renderpass
    vk::RenderPass renderpass = GetRenderPass(info.color_attachment, info.depth_attachment);

    // Find an owner first
    const u64 layout_hash = Common::ComputeHash64(&info.layout, sizeof(PipelineLayoutInfo));
    if (auto iter = pipeline_owners.find(layout_hash); iter != pipeline_owners.end()) {
        return pool_manager.Allocate<Pipeline>(instance, scheduler, pool_manager, *iter->second.get(), type,
                                               info, renderpass, pipeline_cache);
    }

    // Create the layout
    auto result = pipeline_owners.emplace(layout_hash, std::make_unique<PipelineOwner>(instance, info.layout));
    return pool_manager.Allocate<Pipeline>(instance, scheduler, pool_manager, *result.first->second.get(), type,
                                           info, renderpass, pipeline_cache);
}

SamplerHandle Backend::CreateSampler(SamplerInfo info) {
    return pool_manager.Allocate<Sampler>(instance, pool_manager, info);
}

ShaderHandle Backend::CreateShader(ShaderStage stage, std::string_view name, std::string source) {
    return pool_manager.Allocate<Shader>(instance, pool_manager, stage, name, std::move(source));
}

void Backend::BindVertexBuffer(BufferHandle buffer, std::span<const u64> offsets) {
    const Buffer* vertex = static_cast<const Buffer*>(buffer.Get());

    const u32 buffer_count = static_cast<u32>(offsets.size());
    std::array<vk::Buffer, 16> buffers;
    buffers.fill(vertex->GetHandle());

    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.bindVertexBuffers(0, buffer_count, buffers.data(), offsets.data());
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
    for (u32 i = 0; i < set_count; i++) {
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
                                                   reinterpret_cast<const void*>(pipeline_owner.GetData(i)));

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
    /*if (draw_framebuffer == current_framebuffer && renderpass_active) {
        return;
    }*/

    Framebuffer* framebuffer = static_cast<Framebuffer*>(draw_framebuffer.Get());
    //current_framebuffer = draw_framebuffer;

    u32 clear_value_count = 0;
    std::array<vk::ClearValue, 2> clear_values{};

    if (framebuffer->GetColorAttachment().IsValid()) {
        for (int i = 0; i < 4; i++) {
            clear_values[clear_value_count].color.float32[i] = framebuffer->clear_color_value[i];
        }
        clear_value_count++;
    }

    if (framebuffer->GetDepthStencilAttachment().IsValid()) {
        clear_values[clear_value_count].depthStencil.depth = framebuffer->clear_depth_value;
        clear_values[clear_value_count++].depthStencil.stencil = framebuffer->clear_stencil_value;
    }

    // Transition attachments to required layout
    framebuffer->PrepareAttachments();

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
    /*if (renderpass_active) {
        command_buffer.endRenderPass();
    }
    */
    command_buffer.beginRenderPass(renderpass_begin, vk::SubpassContents::eInline);
    //renderpass_active = true;
}

void Backend::OnCommandSwitch(u32 new_slot) {
    // Reset the descriptor pool assigned to the new command slot. This is called
    // after Synchronize, so it's guaranteed that the descriptor sets are no longer
    // in use.
    vk::Device device = instance.GetDevice();
    device.resetDescriptorPool(descriptor_pools[new_slot]);

    // Mark all descriptor sets as dirty
    for (auto& [key, owner] : pipeline_owners) {
        owner->descriptor_dirty.fill(true);
    }
}

} // namespace VideoCore::Vulkan
