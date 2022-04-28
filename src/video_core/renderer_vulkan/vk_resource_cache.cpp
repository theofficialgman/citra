// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_vulkan/vk_resource_cache.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include <algorithm>
#include <array>
#include <type_traits>

namespace Vulkan {

VKResourceCache::~VKResourceCache()
{
    for (int i = 0; i < DESCRIPTOR_SET_LAYOUT_COUNT; i++) {
        g_vk_instace->GetDevice().destroyDescriptorSetLayout(descriptor_layouts[i]);
    }
}

bool VKResourceCache::Initialize()
{
    // Define the descriptor sets we will be using
    std::array<vk::DescriptorSetLayoutBinding, 2> ubo_set = {{
        { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex |
          vk::ShaderStageFlagBits::eGeometry | vk::ShaderStageFlagBits::eFragment }, // shader_data
        { 1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex } // pica_uniforms
    }};

    std::array<vk::DescriptorSetLayoutBinding, 4> texture_set = {{
        { 0, vk::DescriptorType::eSampledImage, 1, vk::ShaderStageFlagBits::eFragment }, // tex0
        { 1, vk::DescriptorType::eSampledImage, 1, vk::ShaderStageFlagBits::eFragment }, // tex1
        { 2, vk::DescriptorType::eSampledImage, 1, vk::ShaderStageFlagBits::eFragment }, // tex2
        { 3, vk::DescriptorType::eSampledImage, 1, vk::ShaderStageFlagBits::eFragment }, // tex_cube
    }};

    std::array<vk::DescriptorSetLayoutBinding, 3> lut_set = {{
        { 0, vk::DescriptorType::eStorageTexelBuffer, 1, vk::ShaderStageFlagBits::eFragment }, // texture_buffer_lut_lf
        { 1, vk::DescriptorType::eStorageTexelBuffer, 1, vk::ShaderStageFlagBits::eFragment }, // texture_buffer_lut_rg
        { 2, vk::DescriptorType::eStorageTexelBuffer, 1, vk::ShaderStageFlagBits::eFragment } // texture_buffer_lut_rgba
    }};

    // Create and store descriptor set layouts
    std::array<vk::DescriptorSetLayoutCreateInfo, DESCRIPTOR_SET_LAYOUT_COUNT> create_infos = {{
        { vk::DescriptorSetLayoutCreateFlags(), ubo_set },
        { vk::DescriptorSetLayoutCreateFlags(), texture_set },
        { vk::DescriptorSetLayoutCreateFlags(), lut_set }
    }};

    for (int i = 0; i < DESCRIPTOR_SET_LAYOUT_COUNT; i++) {
        descriptor_layouts[i] = g_vk_instace->GetDevice().createDescriptorSetLayout(create_infos[i]);
    }

    // Create the standard descriptor set layout
    vk::PipelineLayoutCreateInfo layout_info({}, descriptor_layouts);
    pipeline_layout = g_vk_instace->GetDevice().createPipelineLayoutUnique(layout_info);

    if (!CreateStaticSamplers())
        return false;

    // Create global texture staging buffer
    texture_upload_buffer.Create(MAX_TEXTURE_UPLOAD_BUFFER_SIZE,
                                 vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                 vk::BufferUsageFlagBits::eTransferSrc);

    return true;
}

vk::Sampler VKResourceCache::GetSampler(const SamplerInfo& info)
{
    auto iter = sampler_cache.find(info);
    if (iter != sampler_cache.end()) {
        return iter->second;
    }

    // Create texture sampler
    auto properties = g_vk_instace->GetPhysicalDevice().getProperties();
    auto features = g_vk_instace->GetPhysicalDevice().getFeatures();
    vk::SamplerCreateInfo sampler_info
    (
        {},
        info.mag_filter,
        info.min_filter,
        info.mipmap_mode,
        info.wrapping[0], info.wrapping[1], info.wrapping[2],
        {},
        features.samplerAnisotropy,
        properties.limits.maxSamplerAnisotropy,
        false,
        vk::CompareOp::eAlways,
        {},
        {},
        vk::BorderColor::eFloatTransparentBlack,
        false
    );

    auto sampler = g_vk_instace->GetDevice().createSamplerUnique(sampler_info);
    vk::Sampler handle = sampler.get();

    // Store it even if it failed
    sampler_cache.emplace(info, std::move(sampler));
    return handle;
}

vk::RenderPass VKResourceCache::GetRenderPass(vk::Format color_format, vk::Format depth_format,
                                              u32 multisamples, vk::AttachmentLoadOp load_op)
{
    auto key = std::tie(color_format, depth_format, multisamples, load_op);
    auto it = render_pass_cache.find(key);
    if (it != render_pass_cache.end()) {
        return it->second;
    }

    vk::SubpassDescription subpass({}, vk::PipelineBindPoint::eGraphics);
    std::array<vk::AttachmentDescription, 2> attachments;
    std::array<vk::AttachmentReference, 2> references;
    u32 index = 0;

    if (color_format != vk::Format::eUndefined) {
        references[index] = vk::AttachmentReference{index, vk::ImageLayout::eColorAttachmentOptimal};
        attachments[index] =
        {
            {},
            color_format,
            static_cast<vk::SampleCountFlagBits>(multisamples),
            load_op,
            vk::AttachmentStoreOp::eStore,
            vk::AttachmentLoadOp::eDontCare,
            vk::AttachmentStoreOp::eDontCare,
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::eColorAttachmentOptimal
        };

        subpass.setColorAttachmentCount(1);
        subpass.setPColorAttachments(&references[index++]);
    }

    if (depth_format != vk::Format::eUndefined) {
        references[index] = vk::AttachmentReference{index, vk::ImageLayout::eDepthStencilAttachmentOptimal};
        attachments[index] =
        {
            {},
            depth_format,
            static_cast<vk::SampleCountFlagBits>(multisamples),
            load_op,
            vk::AttachmentStoreOp::eStore,
            vk::AttachmentLoadOp::eDontCare,
            vk::AttachmentStoreOp::eDontCare,
            vk::ImageLayout::eDepthStencilAttachmentOptimal,
            vk::ImageLayout::eDepthStencilAttachmentOptimal
        };

        subpass.setPDepthStencilAttachment(&references[index++]);
    }

    std::array<vk::SubpassDescription, 1> subpasses = { subpass };
    vk::RenderPassCreateInfo renderpass_info({}, attachments, subpasses);

    auto renderpass = g_vk_instace->GetDevice().createRenderPassUnique(renderpass_info);
    vk::RenderPass handle = renderpass.get();

    render_pass_cache.emplace(key, std::move(renderpass));
    return handle;
}
}  // namespace Vulkan
