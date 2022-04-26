#pragma once
#include "vk_buffer.h"
#include <memory>
#include <span>

class VkContext;

class VkTexture : public NonCopyable, public Resource
{
    friend class VkContext;
public:
    VkTexture(const std::shared_ptr<VkContext>& context);
    ~VkTexture() = default;

    void create(int width, int height, vk::ImageType type, vk::Format format = vk::Format::eR8G8B8A8Uint);
    void copy_pixels(std::span<u8> pixels);

private:
    void transition_layout(vk::ImageLayout old_layout, vk::ImageLayout new_layout);

private:
    // Texture buffer
    void* pixels = nullptr;
    std::shared_ptr<VkContext> context;
    uint32_t width = 0, height = 0, channels = 0;
    Buffer staging;

    // Texture objects
    vk::UniqueImage texture;
    vk::UniqueImageView texture_view;
    vk::UniqueDeviceMemory texture_memory;
    vk::UniqueSampler texture_sampler;
    vk::Format format;
};
