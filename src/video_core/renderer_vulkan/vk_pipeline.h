// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include "video_core/common/pipeline.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {
class PoolManager;
}

namespace VideoCore::Vulkan {

class Instance;
class CommandScheduler;

union DescriptorData {
    vk::DescriptorImageInfo image_info;
    vk::DescriptorBufferInfo buffer_info;
    vk::BufferView buffer_view;
};

/**
 * Stores the pipeline layout as well as the descriptor set layouts
 * and update templates associated with those layouts.
 * Functions as the "parent" to a group of pipelines that share the same layout
 */
class PipelineOwner {
    friend class Backend;
public:
    PipelineOwner(Instance& instance, PipelineLayoutInfo info);
    ~PipelineOwner();

    // Disable copy constructor
    PipelineOwner(const PipelineOwner&) = delete;
    PipelineOwner& operator=(const PipelineOwner&) = delete;

    // Assigns data to a particular binding
    void SetBinding(u32 set, u32 binding, DescriptorData data) {
        if (std::memcmp(&update_data[set][binding], &data, sizeof(DescriptorData)) != 0) {
            update_data[set][binding] = data;
            descriptor_dirty[set] = true;
        }
    }

    // Returns the number of descriptor set layouts
    u32 GetDescriptorSetLayoutCount() const {
        return set_layout_count;
    }

    // Returns the most current descriptor update data
    DescriptorData* GetData(u32 set) {
        return update_data.at(set).data();
    }

    vk::DescriptorSetLayout* GetDescriptorSetLayouts() {
        return set_layouts.data();
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
    std::array<vk::DescriptorSet, MAX_BINDING_GROUPS> descriptor_bank;

    // Update data for the descriptor sets
    using SetData = std::array<DescriptorData, MAX_BINDINGS_IN_GROUP>;
    std::array<SetData, MAX_BINDING_GROUPS> update_data{};
    std::array<bool, MAX_BINDING_GROUPS> descriptor_dirty{true};
};

class Pipeline : public VideoCore::PipelineBase {
public:
    Pipeline(Instance& instance, CommandScheduler& scheduler, PoolManager& pool_manager, PipelineOwner& owner,
             PipelineType type, PipelineInfo info, vk::RenderPass renderpass, vk::PipelineCache cache);
    ~Pipeline() override;

    void Free() override;

    void BindTexture(u32 group, u32 slot, TextureHandle handle) override;
    void BindBuffer(u32 group, u32 slot, BufferHandle handle,
                            u32 offset = 0, u32 range = WHOLE_SIZE, u32 view = 0) override;
    void BindSampler(u32 group, u32 slot, SamplerHandle handle) override;
    void BindPushConstant(std::span<const std::byte> data) override;

    void SetViewport(float x, float y, float width, float height) override;
    void SetScissor(s32 x, s32 y, u32 width, u32 height) override;

    void ApplyDynamic(const PipelineInfo& info) override;

    /// Returns the layout tracker that owns this pipeline
    PipelineOwner& GetOwner() const {
        return owner;
    }

    /// Returns the underlying vulkan pipeline handle
    vk::Pipeline GetHandle() const {
        return pipeline;
    }

private:
    Instance& instance;
    CommandScheduler& scheduler;
    PoolManager& pool_manager;

    // The owner of this pipeline
    PipelineOwner& owner;
    vk::Pipeline pipeline;
};

} // namespace VideoCore::Vulkan
