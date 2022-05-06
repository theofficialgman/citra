// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_vulkan/vk_state.h"
#include "video_core/renderer_vulkan/vk_task_scheduler.h"
#include "video_core/renderer_vulkan/vk_resource_cache.h"

namespace Vulkan {

std::unique_ptr<VulkanState> g_vk_state;

// Define bitwise operators for DirtyState enum
DirtyState operator |=(DirtyState lhs, DirtyState rhs) {
    return static_cast<DirtyState> (
        static_cast<unsigned>(lhs) |
        static_cast<unsigned>(rhs)
    );
}

bool operator &(DirtyState lhs, DirtyState rhs) {
    return static_cast<unsigned>(lhs) &
           static_cast<unsigned>(rhs);
}

void VulkanState::Create() {
    // Create a dummy texture which can be used in place of a real binding.
    VKTexture::Info info = {
        .width = 1,
        .height = 1,
        .format = vk::Format::eR8G8B8A8Unorm,
        .type = vk::ImageType::e2D,
        .view_type = vk::ImageViewType::e2D
    };

    dummy_texture.Create(info);
    dummy_texture.TransitionLayout(vk::ImageLayout::eShaderReadOnlyOptimal, g_vk_task_scheduler->GetCommandBuffer());

    // Create descriptor pool
    // TODO: Choose sizes more wisely
    const std::array<vk::DescriptorPoolSize, 3> pool_sizes{{
        { vk::DescriptorType::eUniformBuffer, 32 },
        { vk::DescriptorType::eCombinedImageSampler, 32 },
        { vk::DescriptorType::eStorageTexelBuffer, 32 },
    }};

    auto& device = g_vk_instace->GetDevice();
    vk::DescriptorPoolCreateInfo pool_create_info({}, 1024, pool_sizes);
    desc_pool = device.createDescriptorPoolUnique(pool_create_info);

    // Create descriptor sets
    auto& layouts = g_vk_res_cache->GetDescriptorLayouts();
    vk::DescriptorSetAllocateInfo alloc_info(desc_pool.get(), layouts);
    descriptor_sets = device.allocateDescriptorSetsUnique(alloc_info);

    dirty_flags |= DirtyState::All;
}

void VulkanState::SetVertexBuffer(VKBuffer* buffer, vk::DeviceSize offset) {
    if (vertex_buffer == buffer) {
        return;
    }

    vertex_buffer = buffer;
    vertex_offset = offset;
    dirty_flags |= DirtyState::VertexBuffer;
}

void VulkanState::SetUniformBuffer(UniformID id, VKBuffer* buffer, u32 offset, u32 size) {
    u32 index = static_cast<u32>(id);
    auto& binding = bindings.ubo[index];
    if (binding.buffer != buffer->GetBuffer() || binding.range != size)
    {
        binding.buffer = buffer->GetBuffer();
        binding.range = size;
        dirty_flags |= DirtyState::Uniform;
        bindings.ubo_update[index] = true;
    }
}

void VulkanState::SetTexture(TextureID id, VKTexture* texture) {
    u32 index = static_cast<u32>(id);
    if (bindings.texture[index].imageView == texture->GetView()) {
        return;
    }

    bindings.texture[index].imageView = texture->GetView();
    bindings.texture[index].imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    dirty_flags |= DirtyState::Texture;
    bindings.texture_update[index] = true;
}

void VulkanState::SetTexelBuffer(TexelBufferID id, VKBuffer* buffer) {
    u32 index = static_cast<u32>(id);
    if (bindings.lut[index].buffer == buffer->GetBuffer()) {
        return;
    }

    bindings.lut[index].buffer = buffer->GetBuffer();
    dirty_flags |= DirtyState::TexelBuffer;
    bindings.lut_update[index] = true;
}

void VulkanState::UnbindTexture(VKTexture* image) {
    // Search the texture bindings for the view
    // and replace it with the dummy texture if found
    for (auto& it : bindings.texture) {
        if (it.imageView == image->GetView()) {
            it.imageView = dummy_texture.GetView();
            it.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        }
    }
}

void VulkanState::SetAttachments(VKTexture* color, VKTexture* depth_stencil) {
    color_attachment = color;
    depth_attachment = depth_stencil;
}

void VulkanState::SetRenderArea(vk::Rect2D new_render_area) {
    render_area = new_render_area;
}

void VulkanState::BeginRendering() {
    if (rendering) {
        return;
    }

    // Make sure attachments are in optimal layout
    auto command_buffer = g_vk_task_scheduler->GetCommandBuffer();
    if (color_attachment->GetLayout() != vk::ImageLayout::eColorAttachmentOptimal) {
        color_attachment->TransitionLayout(vk::ImageLayout::eColorAttachmentOptimal, command_buffer);
    }

    if (depth_attachment->GetLayout() != vk::ImageLayout::eDepthStencilAttachmentOptimal) {
        depth_attachment->TransitionLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal, command_buffer);
    }

    // Begin rendering
    vk::RenderingAttachmentInfoKHR color_info(color_attachment->GetView(), color_attachment->GetLayout());
    vk::RenderingAttachmentInfoKHR depth_stencil_info(depth_attachment->GetView(), depth_attachment->GetLayout());

    vk::RenderingInfo render_info
    (
        {}, render_area, 1, {},
        color_info,
        &depth_stencil_info,
        &depth_stencil_info
    );

    command_buffer.beginRendering(render_info);
    rendering = true;
}

void VulkanState::EndRendering() {
    if (!rendering) {
        return;
    }

    auto command_buffer = g_vk_task_scheduler->GetCommandBuffer();
    command_buffer.endRendering();
    rendering = false;
}

void VulkanState::SetViewport(vk::Viewport new_viewport) {
    if (new_viewport == viewport) {
        return;
    }

    viewport = new_viewport;
    dirty_flags |= DirtyState::Viewport;
}

void VulkanState::SetScissor(vk::Rect2D new_scissor) {
    if (new_scissor == scissor) {
        return;
    }

    scissor = new_scissor;
    dirty_flags |= DirtyState::Scissor;
}

void VulkanState::Apply() {
    // Update resources in descriptor sets if changed
    UpdateDescriptorSet();

    // Start rendering if not already started
    BeginRendering();

    // Re-apply dynamic parts of the pipeline
    auto command_buffer = g_vk_task_scheduler->GetCommandBuffer();
    if (dirty_flags & DirtyState::VertexBuffer) {
        command_buffer.bindVertexBuffers(0, vertex_buffer->GetBuffer(), vertex_offset);
    }

    if (dirty_flags & DirtyState::IndexBuffer) {
        command_buffer.bindIndexBuffer(index_buffer->GetBuffer(), index_offset, vk::IndexType::eUint16);
    }

    if (dirty_flags & DirtyState::Viewport) {
        command_buffer.setViewport(0, viewport);
    }

    if (dirty_flags & DirtyState::Scissor) {
        command_buffer.setScissor(0, scissor);
    }

    dirty_flags = DirtyState::None;
}

void VulkanState::UpdateDescriptorSet() {
    std::vector<vk::WriteDescriptorSet> writes;
    auto& device = g_vk_instace->GetDevice();

    // Check if any resource has been updated
    if (dirty_flags & DirtyState::Uniform) {
        for (int i = 0; i < 2; i++) {
            if (bindings.ubo_update[i]) {
                writes.emplace_back(descriptor_sets[i].get(), i, 0, 1, vk::DescriptorType::eUniformBuffer,
                                    nullptr, &bindings.ubo[i]);
                bindings.ubo_update[i] = false;
            }
        }
    }

    if (dirty_flags & DirtyState::Texture) {
        for (int i = 0; i < 4; i++) {
            if (bindings.texture_update[i]) {
                writes.emplace_back(descriptor_sets[i].get(), i, 0, 1, vk::DescriptorType::eCombinedImageSampler,
                                    nullptr, &bindings.texture[i]);
                bindings.texture_update[i] = false;
            }
        }
    }

    if (dirty_flags & DirtyState::TexelBuffer) {
        for (int i = 0; i < 3; i++) {
            if (bindings.lut_update[i]) {
                writes.emplace_back(descriptor_sets[i].get(), i, 0, 1, vk::DescriptorType::eStorageTexelBuffer,
                                    nullptr, &bindings.lut[i]);
                bindings.lut_update[i] = false;
            }
        }
    }

    if (!writes.empty()) {
        device.updateDescriptorSets(writes, {});
    }
}

}  // namespace Vulkan
