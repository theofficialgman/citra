// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include "video_core/common/pipeline.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore::Vulkan {

class Instance;
class CommandScheduler;

union DescriptorData {
    vk::DescriptorImageInfo image_info{};
    vk::DescriptorBufferInfo buffer_info;
    vk::BufferView buffer_view;
};

/**
 * Stores the pipeline layout as well as the descriptor set layouts
 * and update templates associated with those layouts.
 * Functions as the "parent" to a group of pipelines that share the same layout
 */
class PipelineLayout {
public:
    PipelineLayout(Instance& instance, PipelineLayoutInfo info);
    ~PipelineLayout();

    // Disable copy constructor
    PipelineLayout(const PipelineLayout&) = delete;
    PipelineLayout& operator=(const PipelineLayout&) = delete;

    // Assigns data to a particular binding
    void SetBinding(u32 set, u32 binding, DescriptorData data) {
        update_data[set][binding] = data;
    }

    // Returns the most current descriptor update data
    std::span<DescriptorData> GetData(u32 set) {
        return std::span{update_data.at(set).data(), set_layout_count};
    }

    // Returns the underlying vulkan pipeline layout handle
    vk::PipelineLayout GetLayout() const {
        return pipeline_layout;
    }

    // Returns the descriptor set update template handle associated with the provided set index
    vk::DescriptorUpdateTemplate GetUpdateTemplate(u32 set) const {
        return update_templates.at(set);
    }

private:
    Instance& instance;
    vk::PipelineLayout pipeline_layout = VK_NULL_HANDLE;
    u32 set_layout_count = 0;
    std::array<vk::DescriptorSetLayout, MAX_BINDING_GROUPS> set_layouts;
    std::array<vk::DescriptorUpdateTemplate, MAX_BINDING_GROUPS> update_templates;

    // Update data for the descriptor sets
    using SetData = std::array<DescriptorData, MAX_BINDINGS_IN_GROUP>;
    std::array<SetData, MAX_BINDING_GROUPS> update_data;
};

class Pipeline : public VideoCore::PipelineBase {
public:
    Pipeline(Instance& instance, PipelineLayout& owner,
             PipelineType type, PipelineInfo info, vk::PipelineCache cache);
    ~Pipeline() override;

    void BindTexture(u32 group, u32 slot, TextureHandle handle) override;

    void BindBuffer(u32 group, u32 slot, BufferHandle handle, u32 view = 0) override;

    void BindSampler(u32 group, u32 slot, SamplerHandle handle) override;

    /// Returns the layout tracker that owns this pipeline
    PipelineLayout& GetOwner() const {
        return owner;
    }

    /// Returns the underlying vulkan pipeline handle
    vk::Pipeline GetHandle() const {
        return pipeline;
    }

private:
    Instance& instance;
    PipelineLayout& owner;
    vk::Pipeline pipeline;
};

} // namespace VideoCore::Vulkan
