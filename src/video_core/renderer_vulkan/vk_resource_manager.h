// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <utility>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>
#include "common/common_types.h"

namespace Vulkan {

class VKContext;

struct VertexInfo
{
    VertexInfo() = default;
    VertexInfo(glm::vec3 position, glm::vec3 color, glm::vec2 coords) :
        position(position), color(color), texcoords(coords) {};

    glm::vec3 position;
    glm::vec3 color;
    glm::vec2 texcoords;
};

struct Vertex : public VertexInfo
{
    Vertex() = default;
    Vertex(glm::vec3 position, glm::vec3 color = {}, glm::vec2 coords = {}) : VertexInfo(position, color, coords) {};
    static constexpr auto binding_desc = vk::VertexInputBindingDescription(0, sizeof(VertexInfo));
    static constexpr std::array<vk::VertexInputAttributeDescription, 3> attribute_desc =
    {
          vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(VertexInfo, position)),
          vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(VertexInfo, color)),
          vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(VertexInfo, texcoords)),
    };
};

class VKBuffer : public NonCopyable, public Resource
{
    friend class VertexBuffer;
public:
    VKBuffer(std::shared_ptr<VKContext> context);
    ~VKBuffer();

    void Create(uint32_t size, vk::MemoryPropertyFlags properties, vk::BufferUsageFlags usage);
    void Bind(vk::CommandBuffer& command_buffer);

    static uint32_t FindMemoryType(uint32_t type_filter, vk::MemoryPropertyFlags properties, std::shared_ptr<VKContext> context);
    static void CopyBuffer(VKBuffer& src_buffer, VKBuffer& dst_buffer, const vk::BufferCopy& region);

public:
    void* memory = nullptr;
    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory buffer_memory;
    vk::UniqueBufferView buffer_view;
    uint32_t size = 0;

protected:
    std::shared_ptr<VKContext> context;
};

class VKTexture : public NonCopyable, public Resource
{
    friend class VkContext;
public:
    VKTexture(const std::shared_ptr<VKContext>& context);
    ~VKTexture() = default;

    void Create(int width, int height, vk::ImageType type, vk::Format format = vk::Format::eR8G8B8A8Uint);
    void CopyPixels(uint8_t* pixels, uint32_t count);

private:
    void TransitionLayout(vk::ImageLayout old_layout, vk::ImageLayout new_layout);

private:
    // Texture buffer
    void* pixels = nullptr;
    std::shared_ptr<VKContext> context;
    uint32_t width = 0, height = 0, channels = 0;
    VKBuffer staging;

    // Texture objects
    vk::UniqueImage texture;
    vk::UniqueImageView texture_view;
    vk::UniqueDeviceMemory texture_memory;
    vk::UniqueSampler texture_sampler;
    vk::Format format;
};

class OGLShader : private NonCopyable {
public:
    OGLShader() = default;

    OGLShader(OGLShader&& o) noexcept : handle(std::exchange(o.handle, 0)) {}

    ~OGLShader() {
        Release();
    }

    OGLShader& operator=(OGLShader&& o) noexcept {
        Release();
        handle = std::exchange(o.handle, 0);
        return *this;
    }

    void Create(const char* source, GLenum type);

    void Release();

    GLuint handle = 0;
};

class OGLProgram : private NonCopyable {
public:
    OGLProgram() = default;

    OGLProgram(OGLProgram&& o) noexcept : handle(std::exchange(o.handle, 0)) {}

    ~OGLProgram() {
        Release();
    }

    OGLProgram& operator=(OGLProgram&& o) noexcept {
        Release();
        handle = std::exchange(o.handle, 0);
        return *this;
    }

    /// Creates a new program from given shader objects
    void Create(bool separable_program, const std::vector<GLuint>& shaders);

    /// Creates a new program from given shader soruce code
    void Create(const char* vert_shader, const char* frag_shader);

    /// Deletes the internal OpenGL resource
    void Release();

    GLuint handle = 0;
};

class OGLPipeline : private NonCopyable {
public:
    OGLPipeline() = default;
    OGLPipeline(OGLPipeline&& o) noexcept {
        handle = std::exchange<GLuint>(o.handle, 0);
    }
    ~OGLPipeline() {
        Release();
    }
    OGLPipeline& operator=(OGLPipeline&& o) noexcept {
        Release();
        handle = std::exchange<GLuint>(o.handle, 0);
        return *this;
    }

    /// Creates a new internal OpenGL resource and stores the handle
    void Create();

    /// Deletes the internal OpenGL resource
    void Release();

    GLuint handle = 0;
};

class OGLBuffer : private NonCopyable {
public:
    OGLBuffer() = default;

    OGLBuffer(OGLBuffer&& o) noexcept : handle(std::exchange(o.handle, 0)) {}

    ~OGLBuffer() {
        Release();
    }

    OGLBuffer& operator=(OGLBuffer&& o) noexcept {
        Release();
        handle = std::exchange(o.handle, 0);
        return *this;
    }

    /// Creates a new internal OpenGL resource and stores the handle
    void Create();

    /// Deletes the internal OpenGL resource
    void Release();

    GLuint handle = 0;
};

class OGLVertexArray : private NonCopyable {
public:
    OGLVertexArray() = default;

    OGLVertexArray(OGLVertexArray&& o) noexcept : handle(std::exchange(o.handle, 0)) {}

    ~OGLVertexArray() {
        Release();
    }

    OGLVertexArray& operator=(OGLVertexArray&& o) noexcept {
        Release();
        handle = std::exchange(o.handle, 0);
        return *this;
    }

    /// Creates a new internal OpenGL resource and stores the handle
    void Create();

    /// Deletes the internal OpenGL resource
    void Release();

    GLuint handle = 0;
};

class OGLFramebuffer : private NonCopyable {
public:
    OGLFramebuffer() = default;

    OGLFramebuffer(OGLFramebuffer&& o) noexcept : handle(std::exchange(o.handle, 0)) {}

    ~OGLFramebuffer() {
        Release();
    }

    OGLFramebuffer& operator=(OGLFramebuffer&& o) noexcept {
        Release();
        handle = std::exchange(o.handle, 0);
        return *this;
    }

    /// Creates a new internal OpenGL resource and stores the handle
    void Create();

    /// Deletes the internal OpenGL resource
    void Release();

    GLuint handle = 0;
};

} // namespace OpenGL
