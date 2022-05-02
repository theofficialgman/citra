// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_vulkan/vk_state.h"

namespace Vulkan {

std::unique_ptr<VulkanState> g_vk_state;

// Define bitwise operators for DirtyState enum
DirtyState operator |=(DirtyState lhs, DirtyState rhs) {
    return static_cast<DirtyState> (
        static_cast<unsigned>(lhs) |
        static_cast<unsigned>(rhs)
    );
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
    //dummy_texture.TransitionLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

    dirty_flags |= DirtyState::All;
}

void VulkanState::SetVertexBuffer(VKBuffer* buffer, vk::DeviceSize offset) {
    if (vertex_buffer == buffer) {
        return;
    }

    vertex_buffer = buffer;
    vertex_buffer_offset = offset;
    dirty_flags |= DirtyState::VertexBuffer;
}

void VulkanState::SetFramebuffer(VKFramebuffer* buffer) {
    // Should not be changed within a render pass.
    //ASSERT(!InRenderPass());
    //framebuffer = buffer;
}

void VulkanState::SetPipeline(const VKPipeline* new_pipeline) {
    if (new_pipeline == pipeline)
        return;

    pipeline = new_pipeline;
    dirty_flags |= DirtyState::Pipeline;
}

void VulkanState::SetUniformBuffer(UniformID id, VKBuffer* buffer, u32 offset, u32 size) {
    auto& binding = bindings.ubo[static_cast<u32>(id)];
    if (binding.buffer != buffer->GetBuffer() || binding.range != size)
    {
        binding.buffer = buffer->GetBuffer();
        binding.range = size;
        dirty_flags |= DirtyState::Uniform;
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
}

void VulkanState::SetTexelBuffer(TexelBufferID id, VKBuffer* buffer) {
    u32 index = static_cast<u32>(id);
    if (bindings.lut[index].buffer == buffer->GetBuffer()) {
        return;
    }

    bindings.lut[index].buffer = buffer->GetBuffer();
    dirty_flags |= DirtyState::TexelBuffer;
}

void VulkanState::SetImageTexture(VKTexture* image) {
    // TODO
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

void VulkanState::BeginRenderPass()
{
  if (InRenderPass())
    return;

  m_current_render_pass = m_framebuffer->GetLoadRenderPass();
  m_framebuffer_render_area = m_framebuffer->GetRect();

  VkRenderPassBeginInfo begin_info = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                                      nullptr,
                                      m_current_render_pass,
                                      m_framebuffer->GetFB(),
                                      m_framebuffer_render_area,
                                      0,
                                      nullptr};

  vkCmdBeginRenderPass(g_command_buffer_mgr->GetCurrentCommandBuffer(), &begin_info,
                       VK_SUBPASS_CONTENTS_INLINE);
}

void StateTracker::BeginDiscardRenderPass()
{
  if (InRenderPass())
    return;

  m_current_render_pass = m_framebuffer->GetDiscardRenderPass();
  m_framebuffer_render_area = m_framebuffer->GetRect();

  VkRenderPassBeginInfo begin_info = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                                      nullptr,
                                      m_current_render_pass,
                                      m_framebuffer->GetFB(),
                                      m_framebuffer_render_area,
                                      0,
                                      nullptr};

  vkCmdBeginRenderPass(g_command_buffer_mgr->GetCurrentCommandBuffer(), &begin_info,
                       VK_SUBPASS_CONTENTS_INLINE);
}

void StateTracker::EndRenderPass()
{
  if (!InRenderPass())
    return;

  vkCmdEndRenderPass(g_command_buffer_mgr->GetCurrentCommandBuffer());
  m_current_render_pass = VK_NULL_HANDLE;
}

void StateTracker::BeginClearRenderPass(const VkRect2D& area, const VkClearValue* clear_values,
                                        u32 num_clear_values)
{
  ASSERT(!InRenderPass());

  m_current_render_pass = m_framebuffer->GetClearRenderPass();
  m_framebuffer_render_area = area;

  VkRenderPassBeginInfo begin_info = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                                      nullptr,
                                      m_current_render_pass,
                                      m_framebuffer->GetFB(),
                                      m_framebuffer_render_area,
                                      num_clear_values,
                                      clear_values};

  vkCmdBeginRenderPass(g_command_buffer_mgr->GetCurrentCommandBuffer(), &begin_info,
                       VK_SUBPASS_CONTENTS_INLINE);
}

void StateTracker::SetViewport(const VkViewport& viewport)
{
  if (memcmp(&m_viewport, &viewport, sizeof(viewport)) == 0)
    return;

  m_viewport = viewport;
  m_dirty_flags |= DIRTY_FLAG_VIEWPORT;
}

void StateTracker::SetScissor(const VkRect2D& scissor)
{
  if (memcmp(&m_scissor, &scissor, sizeof(scissor)) == 0)
    return;

  m_scissor = scissor;
  m_dirty_flags |= DIRTY_FLAG_SCISSOR;
}

bool StateTracker::Bind()
{
  // Must have a pipeline.
  if (!m_pipeline)
    return false;

  // Check the render area if we were in a clear pass.
  if (m_current_render_pass == m_framebuffer->GetClearRenderPass() && !IsViewportWithinRenderArea())
    EndRenderPass();

  // Get a new descriptor set if any parts have changed
  if (!UpdateDescriptorSet())
  {
    // We can fail to allocate descriptors if we exhaust the pool for this command buffer.
    WARN_LOG_FMT(VIDEO, "Failed to get a descriptor set, executing buffer");
    Renderer::GetInstance()->ExecuteCommandBuffer(false, false);
    if (!UpdateDescriptorSet())
    {
      // Something strange going on.
      ERROR_LOG_FMT(VIDEO, "Failed to get descriptor set, skipping draw");
      return false;
    }
  }

  // Start render pass if not already started
  if (!InRenderPass())
    BeginRenderPass();

  // Re-bind parts of the pipeline
  const VkCommandBuffer command_buffer = g_command_buffer_mgr->GetCurrentCommandBuffer();
  if (m_dirty_flags & DIRTY_FLAG_VERTEX_BUFFER)
    vkCmdBindVertexBuffers(command_buffer, 0, 1, &m_vertex_buffer, &m_vertex_buffer_offset);

  if (m_dirty_flags & DIRTY_FLAG_INDEX_BUFFER)
    vkCmdBindIndexBuffer(command_buffer, m_index_buffer, m_index_buffer_offset, m_index_type);

  if (m_dirty_flags & DIRTY_FLAG_PIPELINE)
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline->GetVkPipeline());

  if (m_dirty_flags & DIRTY_FLAG_VIEWPORT)
    vkCmdSetViewport(command_buffer, 0, 1, &m_viewport);

  if (m_dirty_flags & DIRTY_FLAG_SCISSOR)
    vkCmdSetScissor(command_buffer, 0, 1, &m_scissor);

  m_dirty_flags &= ~(DIRTY_FLAG_VERTEX_BUFFER | DIRTY_FLAG_INDEX_BUFFER | DIRTY_FLAG_PIPELINE |
                     DIRTY_FLAG_VIEWPORT | DIRTY_FLAG_SCISSOR);
  return true;
}

bool StateTracker::BindCompute()
{
  if (!m_compute_shader)
    return false;

  // Can't kick compute in a render pass.
  if (InRenderPass())
    EndRenderPass();

  const VkCommandBuffer command_buffer = g_command_buffer_mgr->GetCurrentCommandBuffer();
  if (m_dirty_flags & DIRTY_FLAG_COMPUTE_SHADER)
  {
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      m_compute_shader->GetComputePipeline());
  }

  if (!UpdateComputeDescriptorSet())
  {
    WARN_LOG_FMT(VIDEO, "Failed to get a compute descriptor set, executing buffer");
    Renderer::GetInstance()->ExecuteCommandBuffer(false, false);
    if (!UpdateComputeDescriptorSet())
    {
      // Something strange going on.
      ERROR_LOG_FMT(VIDEO, "Failed to get descriptor set, skipping dispatch");
      return false;
    }
  }

  m_dirty_flags &= ~DIRTY_FLAG_COMPUTE_SHADER;
  return true;
}

bool StateTracker::IsWithinRenderArea(s32 x, s32 y, u32 width, u32 height) const
{
  // Check that the viewport does not lie outside the render area.
  // If it does, we need to switch to a normal load/store render pass.
  s32 left = m_framebuffer_render_area.offset.x;
  s32 top = m_framebuffer_render_area.offset.y;
  s32 right = left + static_cast<s32>(m_framebuffer_render_area.extent.width);
  s32 bottom = top + static_cast<s32>(m_framebuffer_render_area.extent.height);
  s32 test_left = x;
  s32 test_top = y;
  s32 test_right = test_left + static_cast<s32>(width);
  s32 test_bottom = test_top + static_cast<s32>(height);
  return test_left >= left && test_right <= right && test_top >= top && test_bottom <= bottom;
}

bool StateTracker::IsViewportWithinRenderArea() const
{
  return IsWithinRenderArea(static_cast<s32>(m_viewport.x), static_cast<s32>(m_viewport.y),
                            static_cast<u32>(m_viewport.width),
                            static_cast<u32>(m_viewport.height));
}

void StateTracker::EndClearRenderPass()
{
  if (m_current_render_pass != m_framebuffer->GetClearRenderPass())
    return;

  // End clear render pass. Bind() will call BeginRenderPass() which
  // will switch to the load/store render pass.
  EndRenderPass();
}

bool StateTracker::UpdateDescriptorSet()
{
  if (m_pipeline->GetUsage() == AbstractPipelineUsage::GX)
    return UpdateGXDescriptorSet();
  else
    return UpdateUtilityDescriptorSet();
}

bool StateTracker::UpdateGXDescriptorSet()
{
  const size_t MAX_DESCRIPTOR_WRITES = NUM_UBO_DESCRIPTOR_SET_BINDINGS +  // UBO
                                       1 +                                // Samplers
                                       1;                                 // SSBO
  std::array<VkWriteDescriptorSet, MAX_DESCRIPTOR_WRITES> writes;
  u32 num_writes = 0;

  if (m_dirty_flags & DIRTY_FLAG_GX_UBOS || m_gx_descriptor_sets[0] == VK_NULL_HANDLE)
  {
    m_gx_descriptor_sets[0] = g_command_buffer_mgr->AllocateDescriptorSet(
        g_object_cache->GetDescriptorSetLayout(DESCRIPTOR_SET_LAYOUT_STANDARD_UNIFORM_BUFFERS));
    if (m_gx_descriptor_sets[0] == VK_NULL_HANDLE)
      return false;

    for (size_t i = 0; i < NUM_UBO_DESCRIPTOR_SET_BINDINGS; i++)
    {
      if (i == UBO_DESCRIPTOR_SET_BINDING_GS &&
          !g_ActiveConfig.backend_info.bSupportsGeometryShaders)
      {
        continue;
      }

      writes[num_writes++] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                              nullptr,
                              m_gx_descriptor_sets[0],
                              static_cast<uint32_t>(i),
                              0,
                              1,
                              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                              nullptr,
                              &m_bindings.gx_ubo_bindings[i],
                              nullptr};
    }

    m_dirty_flags = (m_dirty_flags & ~DIRTY_FLAG_GX_UBOS) | DIRTY_FLAG_DESCRIPTOR_SETS;
  }

  if (m_dirty_flags & DIRTY_FLAG_GX_SAMPLERS || m_gx_descriptor_sets[1] == VK_NULL_HANDLE)
  {
    m_gx_descriptor_sets[1] = g_command_buffer_mgr->AllocateDescriptorSet(
        g_object_cache->GetDescriptorSetLayout(DESCRIPTOR_SET_LAYOUT_STANDARD_SAMPLERS));
    if (m_gx_descriptor_sets[1] == VK_NULL_HANDLE)
      return false;

    writes[num_writes++] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                            nullptr,
                            m_gx_descriptor_sets[1],
                            0,
                            0,
                            static_cast<u32>(NUM_PIXEL_SHADER_SAMPLERS),
                            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            m_bindings.samplers.data(),
                            nullptr,
                            nullptr};
    m_dirty_flags = (m_dirty_flags & ~DIRTY_FLAG_GX_SAMPLERS) | DIRTY_FLAG_DESCRIPTOR_SETS;
  }

  if (g_ActiveConfig.backend_info.bSupportsBBox &&
      (m_dirty_flags & DIRTY_FLAG_GX_SSBO || m_gx_descriptor_sets[2] == VK_NULL_HANDLE))
  {
    m_gx_descriptor_sets[2] =
        g_command_buffer_mgr->AllocateDescriptorSet(g_object_cache->GetDescriptorSetLayout(
            DESCRIPTOR_SET_LAYOUT_STANDARD_SHADER_STORAGE_BUFFERS));
    if (m_gx_descriptor_sets[2] == VK_NULL_HANDLE)
      return false;

    writes[num_writes++] = {
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_gx_descriptor_sets[2], 0,      0, 1,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,      nullptr, &m_bindings.ssbo,        nullptr};
    m_dirty_flags = (m_dirty_flags & ~DIRTY_FLAG_GX_SSBO) | DIRTY_FLAG_DESCRIPTOR_SETS;
  }

  if (num_writes > 0)
    vkUpdateDescriptorSets(g_vulkan_context->GetDevice(), num_writes, writes.data(), 0, nullptr);

  if (m_dirty_flags & DIRTY_FLAG_DESCRIPTOR_SETS)
  {
    vkCmdBindDescriptorSets(g_command_buffer_mgr->GetCurrentCommandBuffer(),
                            VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline->GetVkPipelineLayout(), 0,
                            g_ActiveConfig.backend_info.bSupportsBBox ?
                                NUM_GX_DESCRIPTOR_SETS :
                                (NUM_GX_DESCRIPTOR_SETS - 1),
                            m_gx_descriptor_sets.data(),
                            g_ActiveConfig.backend_info.bSupportsGeometryShaders ?
                                NUM_UBO_DESCRIPTOR_SET_BINDINGS :
                                (NUM_UBO_DESCRIPTOR_SET_BINDINGS - 1),
                            m_bindings.gx_ubo_offsets.data());
    m_dirty_flags &= ~(DIRTY_FLAG_DESCRIPTOR_SETS | DIRTY_FLAG_GX_UBO_OFFSETS);
  }
  else if (m_dirty_flags & DIRTY_FLAG_GX_UBO_OFFSETS)
  {
    vkCmdBindDescriptorSets(g_command_buffer_mgr->GetCurrentCommandBuffer(),
                            VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline->GetVkPipelineLayout(), 0,
                            1, m_gx_descriptor_sets.data(),
                            g_ActiveConfig.backend_info.bSupportsGeometryShaders ?
                                NUM_UBO_DESCRIPTOR_SET_BINDINGS :
                                (NUM_UBO_DESCRIPTOR_SET_BINDINGS - 1),
                            m_bindings.gx_ubo_offsets.data());
    m_dirty_flags &= ~DIRTY_FLAG_GX_UBO_OFFSETS;
  }

  return true;
}

bool StateTracker::UpdateUtilityDescriptorSet()
{
  // Max number of updates - UBO, Samplers, TexelBuffer
  std::array<VkWriteDescriptorSet, 3> dswrites;
  u32 writes = 0;

  // Allocate descriptor sets.
  if (m_dirty_flags & DIRTY_FLAG_UTILITY_UBO || m_utility_descriptor_sets[0] == VK_NULL_HANDLE)
  {
    m_utility_descriptor_sets[0] = g_command_buffer_mgr->AllocateDescriptorSet(
        g_object_cache->GetDescriptorSetLayout(DESCRIPTOR_SET_LAYOUT_UTILITY_UNIFORM_BUFFER));
    if (!m_utility_descriptor_sets[0])
      return false;

    dswrites[writes++] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                          nullptr,
                          m_utility_descriptor_sets[0],
                          0,
                          0,
                          1,
                          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                          nullptr,
                          &m_bindings.utility_ubo_binding,
                          nullptr};

    m_dirty_flags = (m_dirty_flags & ~DIRTY_FLAG_UTILITY_UBO) | DIRTY_FLAG_DESCRIPTOR_SETS;
  }

  if (m_dirty_flags & DIRTY_FLAG_UTILITY_BINDINGS || m_utility_descriptor_sets[1] == VK_NULL_HANDLE)
  {
    m_utility_descriptor_sets[1] = g_command_buffer_mgr->AllocateDescriptorSet(
        g_object_cache->GetDescriptorSetLayout(DESCRIPTOR_SET_LAYOUT_UTILITY_SAMPLERS));
    if (!m_utility_descriptor_sets[1])
      return false;

    dswrites[writes++] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                          nullptr,
                          m_utility_descriptor_sets[1],
                          0,
                          0,
                          NUM_PIXEL_SHADER_SAMPLERS,
                          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                          m_bindings.samplers.data(),
                          nullptr,
                          nullptr};
    dswrites[writes++] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                          nullptr,
                          m_utility_descriptor_sets[1],
                          8,
                          0,
                          1,
                          VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
                          nullptr,
                          nullptr,
                          m_bindings.texel_buffers.data()};

    m_dirty_flags = (m_dirty_flags & ~DIRTY_FLAG_UTILITY_BINDINGS) | DIRTY_FLAG_DESCRIPTOR_SETS;
  }

  if (writes > 0)
    vkUpdateDescriptorSets(g_vulkan_context->GetDevice(), writes, dswrites.data(), 0, nullptr);

  if (m_dirty_flags & DIRTY_FLAG_DESCRIPTOR_SETS)
  {
    vkCmdBindDescriptorSets(g_command_buffer_mgr->GetCurrentCommandBuffer(),
                            VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline->GetVkPipelineLayout(), 0,
                            NUM_UTILITY_DESCRIPTOR_SETS, m_utility_descriptor_sets.data(), 1,
                            &m_bindings.utility_ubo_offset);
    m_dirty_flags &= ~(DIRTY_FLAG_DESCRIPTOR_SETS | DIRTY_FLAG_UTILITY_UBO_OFFSET);
  }
  else if (m_dirty_flags & DIRTY_FLAG_UTILITY_UBO_OFFSET)
  {
    vkCmdBindDescriptorSets(g_command_buffer_mgr->GetCurrentCommandBuffer(),
                            VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline->GetVkPipelineLayout(), 0,
                            1, m_utility_descriptor_sets.data(), 1, &m_bindings.utility_ubo_offset);
    m_dirty_flags &= ~(DIRTY_FLAG_DESCRIPTOR_SETS | DIRTY_FLAG_UTILITY_UBO_OFFSET);
  }

  return true;
}

bool StateTracker::UpdateComputeDescriptorSet()
{
  // Max number of updates - UBO, Samplers, TexelBuffer, Image
  std::array<VkWriteDescriptorSet, 4> dswrites;

  // Allocate descriptor sets.
  if (m_dirty_flags & DIRTY_FLAG_COMPUTE_BINDINGS)
  {
    m_compute_descriptor_set = g_command_buffer_mgr->AllocateDescriptorSet(
        g_object_cache->GetDescriptorSetLayout(DESCRIPTOR_SET_LAYOUT_COMPUTE));
    dswrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                   nullptr,
                   m_compute_descriptor_set,
                   0,
                   0,
                   1,
                   VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                   nullptr,
                   &m_bindings.utility_ubo_binding,
                   nullptr};
    dswrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                   nullptr,
                   m_compute_descriptor_set,
                   1,
                   0,
                   NUM_COMPUTE_SHADER_SAMPLERS,
                   VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                   m_bindings.samplers.data(),
                   nullptr,
                   nullptr};
    dswrites[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                   nullptr,
                   m_compute_descriptor_set,
                   3,
                   0,
                   NUM_COMPUTE_TEXEL_BUFFERS,
                   VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
                   nullptr,
                   nullptr,
                   m_bindings.texel_buffers.data()};
    dswrites[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                   nullptr,
                   m_compute_descriptor_set,
                   5,
                   0,
                   1,
                   VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                   &m_bindings.image_texture,
                   nullptr,
                   nullptr};

    vkUpdateDescriptorSets(g_vulkan_context->GetDevice(), static_cast<uint32_t>(dswrites.size()),
                           dswrites.data(), 0, nullptr);
    m_dirty_flags =
        (m_dirty_flags & ~DIRTY_FLAG_COMPUTE_BINDINGS) | DIRTY_FLAG_COMPUTE_DESCRIPTOR_SET;
  }

  if (m_dirty_flags & DIRTY_FLAG_COMPUTE_DESCRIPTOR_SET)
  {
    vkCmdBindDescriptorSets(g_command_buffer_mgr->GetCurrentCommandBuffer(),
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            g_object_cache->GetPipelineLayout(PIPELINE_LAYOUT_COMPUTE), 0, 1,
                            &m_compute_descriptor_set, 1, &m_bindings.utility_ubo_offset);
    m_dirty_flags &= ~DIRTY_FLAG_COMPUTE_DESCRIPTOR_SET;
  }

  return true;
}

}  // namespace Vulkan
