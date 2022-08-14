// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <bit>
#include "common/alignment.h"
#include "common/microprofile.h"
#include "core/hw/gpu.h"
#include "video_core/pica_state.h"
#include "video_core/common/rasterizer.h"
#include "video_core/common/renderer.h"
#include "video_core/common/pipeline_cache.h"
#include "video_core/video_core.h"

namespace VideoCore {

using PixelFormat = SurfaceParams::PixelFormat;
using SurfaceType = SurfaceParams::SurfaceType;

MICROPROFILE_DEFINE(VertexSetup, "Rasterizer", "Vertex Setup", MP_RGB(255, 128, 0));
MICROPROFILE_DEFINE(VertexShader, "Rasterizer", "Vertex Shader Setup", MP_RGB(192, 128, 128));
MICROPROFILE_DEFINE(GeometryShader, "Rasterizer", "Geometry Shader Setup", MP_RGB(128, 192, 128));
MICROPROFILE_DEFINE(Drawing, "Rasterizer", "Drawing", MP_RGB(128, 128, 192));
MICROPROFILE_DEFINE(Blits, "Rasterizer", "Blits", MP_RGB(100, 100, 255));
MICROPROFILE_DEFINE(CacheManagement, "Rasterizer", "Cache Management", MP_RGB(100, 255, 100));

HardwareVertex::HardwareVertex(const Pica::Shader::OutputVertex& v, bool flip_quaternion) {
    position[0] = v.pos.x.ToFloat32();
    position[1] = v.pos.y.ToFloat32();
    position[2] = v.pos.z.ToFloat32();
    position[3] = v.pos.w.ToFloat32();
    color[0] = v.color.x.ToFloat32();
    color[1] = v.color.y.ToFloat32();
    color[2] = v.color.z.ToFloat32();
    color[3] = v.color.w.ToFloat32();
    tex_coord0[0] = v.tc0.x.ToFloat32();
    tex_coord0[1] = v.tc0.y.ToFloat32();
    tex_coord1[0] = v.tc1.x.ToFloat32();
    tex_coord1[1] = v.tc1.y.ToFloat32();
    tex_coord2[0] = v.tc2.x.ToFloat32();
    tex_coord2[1] = v.tc2.y.ToFloat32();
    tex_coord0_w = v.tc0_w.ToFloat32();
    normquat[0] = v.quat.x.ToFloat32();
    normquat[1] = v.quat.y.ToFloat32();
    normquat[2] = v.quat.z.ToFloat32();
    normquat[3] = v.quat.w.ToFloat32();
    view[0] = v.view.x.ToFloat32();
    view[1] = v.view.y.ToFloat32();
    view[2] = v.view.z.ToFloat32();

    if (flip_quaternion) {
        normquat = -normquat;
    }
}

/**
 * This maps to the following layout in GLSL code:
 *  layout(location = 0) in vec4 vert_position;
 *  layout(location = 1) in vec4 vert_color;
 *  layout(location = 2) in vec2 vert_texcoord0;
 *  layout(location = 3) in vec2 vert_texcoord1;
 *  layout(location = 4) in vec2 vert_texcoord2;
 *  layout(location = 5) in float vert_texcoord0_w;
 *  layout(location = 6) in vec4 vert_normquat;
 *  layout(location = 7) in vec3 vert_view;
 */
constexpr VertexLayout HardwareVertex::GetVertexLayout() {
    VertexLayout layout{};
    layout.attribute_count = 8;
    layout.binding_count = 1;

    // Define binding
    layout.bindings[0].binding.Assign(0);
    layout.bindings[0].fixed.Assign(0);
    layout.bindings[0].stride.Assign(sizeof(HardwareVertex));

    // Define attributes
    constexpr std::array sizes = {4, 4, 2, 2, 2, 1, 4, 3};
    u32 offset = 0;

    for (u32 loc = 0; loc < 8; loc++) {
        VertexAttribute& attribute = layout.attributes[loc];
        attribute.binding.Assign(0);
        attribute.location.Assign(loc);
        attribute.offset.Assign(offset);
        attribute.type.Assign(AttribType::Float);
        attribute.size.Assign(sizes[loc]);
        offset += sizes[loc] * sizeof(float);
    }

    return layout;
}

constexpr u32 UTILITY_GROUP = 0;
constexpr u32 TEXTURE_GROUP = 1;
constexpr u32 SAMPLER_GROUP = 2;

// Rasterizer pipeline layout
constexpr PipelineLayoutInfo RASTERIZER_PIPELINE_LAYOUT = {
    .group_count = 3,
    .binding_groups = {
        // Uniform + LUT set
        BindingGroup{
            BindingType::Uniform,
            BindingType::Uniform,
            BindingType::TexelBuffer,
            BindingType::TexelBuffer,
            BindingType::TexelBuffer
        }, // Texture unit set
        BindingGroup{
            BindingType::Texture,
            BindingType::Texture,
            BindingType::Texture,
            BindingType::Texture
        }, // Texture unit sampler set
        BindingGroup{
            BindingType::Sampler,
            BindingType::Sampler,
            BindingType::Sampler,
            BindingType::Sampler
        }
    }
};

// Define information about the rasterizer buffers
static constexpr BufferInfo VERTEX_BUFFER_INFO = {
    .capacity = 16 * 1024 * 1024,
    .usage = BufferUsage::Vertex
};

static constexpr BufferInfo INDEX_BUFFER_INFO = {
    .capacity = 1 * 1024 * 1024,
    .usage = BufferUsage::Index
};

static constexpr BufferInfo UNIFORM_BUFFER_INFO = {
    .capacity = 2 * 1024 * 1024,
    .usage = BufferUsage::Uniform
};

static constexpr BufferInfo TEXEL_BUFFER_LF_INFO = {
    .capacity = 1 * 1024 * 1024,
    .usage = BufferUsage::Texel,
    .views = {
        ViewFormat::R32G32Float
    }
};

static constexpr BufferInfo TEXEL_BUFFER_INFO = {
    .capacity = 1 * 1024 * 1024,
    .usage = BufferUsage::Texel,
    .views = {
        ViewFormat::R32G32Float,
        ViewFormat::R32G32B32A32Float
    }
};

Rasterizer::Rasterizer(Frontend::EmuWindow& emu_window, std::unique_ptr<BackendBase>& backend) :
    backend(backend), res_cache(backend) {

    // Clipping plane 0 is always enabled for PICA fixed clip plane z <= 0
    //state.clip_distance[0] = true;

    // Set the default vertex buffer layout for the rasterizer pipeline
    raster_info.vertex_layout = HardwareVertex::GetVertexLayout();

    std::array<u8, 4> clear_data = {0, 0, 0, 1};
    const TextureInfo clear_info = {
        .width = 1,
        .height = 1,
        .levels = 1,
        .type = TextureType::Texture2D,
        .view_type = TextureViewType::View2D,
        .format = TextureFormat::RGBA8
    };

    // Create a 1x1 clear texture to use in the NULL case
    clear_texture = backend->CreateTexture(clear_info);
    clear_texture->Upload(Rect2D{0, 0, 1, 1}, 1, clear_data);

    // Create rasterizer buffers
    vertex_buffer = backend->CreateBuffer(VERTEX_BUFFER_INFO);
    index_buffer = backend->CreateBuffer(INDEX_BUFFER_INFO);
    uniform_buffer_vs = backend->CreateBuffer(UNIFORM_BUFFER_INFO);
    uniform_buffer_fs = backend->CreateBuffer(UNIFORM_BUFFER_INFO);
    texel_buffer_lut = backend->CreateBuffer(TEXEL_BUFFER_INFO);
    texel_buffer_lut_lf = backend->CreateBuffer(TEXEL_BUFFER_LF_INFO);

    const SamplerInfo cube_sampler_info = {
        .mag_filter = Pica::TextureFilter::Linear,
        .min_filter = Pica::TextureFilter::Linear,
        .mip_filter = Pica::TextureFilter::Linear,
        .wrap_s = Pica::WrapMode::ClampToEdge,
        .wrap_t = Pica::WrapMode::ClampToEdge
    };

    // TODO: Texture cubes
    texture_cube_sampler = backend->CreateSampler(cube_sampler_info);

    // TODO: Have the backend say this
    uniform_buffer_alignment = 64;
    uniform_size_aligned_vs = Common::AlignUp<std::size_t>(sizeof(VSUniformData), uniform_buffer_alignment);
    uniform_size_aligned_fs = Common::AlignUp<std::size_t>(sizeof(UniformData), uniform_buffer_alignment);

    // Create pipeline cache
    pipeline_cache = std::make_unique<PipelineCache>(emu_window, backend);

    // Initialize the rasterization pipeline info
    raster_info.vertex_layout = HardwareVertex::GetVertexLayout();
    raster_info.layout = RASTERIZER_PIPELINE_LAYOUT;

    // Synchronize pica state
    SyncEntireState();
}

Rasterizer::~Rasterizer() = default;

void Rasterizer::LoadDiskResources(const std::atomic_bool& stop_loading, const DiskLoadCallback& callback) {
    pipeline_cache->LoadDiskCache(stop_loading, callback);
}

void Rasterizer::SyncEntireState() {
    // Sync fixed function state
    SyncClipEnabled();
    SyncCullMode();
    SyncBlendEnabled();
    SyncBlendFuncs();
    SyncBlendColor();
    SyncLogicOp();
    SyncStencilTest();
    SyncDepthTest();
    SyncColorWriteMask();
    SyncStencilWriteMask();
    SyncDepthWriteMask();

    // Sync uniforms
    SyncClipCoef();
    SyncDepthScale();
    SyncDepthOffset();
    SyncAlphaTest();
    SyncCombinerColor();
    auto& tev_stages = Pica::g_state.regs.texturing.GetTevStages();
    for (std::size_t index = 0; index < tev_stages.size(); ++index)
        SyncTevConstColor(index, tev_stages[index]);

    SyncGlobalAmbient();
    for (unsigned light_index = 0; light_index < 8; light_index++) {
        SyncLightSpecular0(light_index);
        SyncLightSpecular1(light_index);
        SyncLightDiffuse(light_index);
        SyncLightAmbient(light_index);
        SyncLightPosition(light_index);
        SyncLightDistanceAttenuationBias(light_index);
        SyncLightDistanceAttenuationScale(light_index);
    }

    SyncFogColor();
    SyncProcTexNoise();
    SyncProcTexBias();
    SyncShadowBias();
    SyncShadowTextureBias();
}

/**
 * This is a helper function to resolve an issue when interpolating opposite quaternions. See below
 * for a detailed description of this issue (yuriks):
 *
 * For any rotation, there are two quaternions Q, and -Q, that represent the same rotation. If you
 * interpolate two quaternions that are opposite, instead of going from one rotation to another
 * using the shortest path, you'll go around the longest path. You can test if two quaternions are
 * opposite by checking if Dot(Q1, Q2) < 0. In that case, you can flip either of them, therefore
 * making Dot(Q1, -Q2) positive.
 *
 * This solution corrects this issue per-vertex before passing the quaternions to OpenGL. This is
 * correct for most cases but can still rotate around the long way sometimes. An implementation
 * which did `lerp(lerp(Q1, Q2), Q3)` (with proper weighting), applying the dot product check
 * between each step would work for those cases at the cost of being more complex to implement.
 *
 * Fortunately however, the 3DS hardware happens to also use this exact same logic to work around
 * these issues, making this basic implementation actually more accurate to the hardware.
 */
static bool AreQuaternionsOpposite(Common::Vec4<Pica::float24> qa, Common::Vec4<Pica::float24> qb) {
    Common::Vec4f a{qa.x.ToFloat32(), qa.y.ToFloat32(), qa.z.ToFloat32(), qa.w.ToFloat32()};
    Common::Vec4f b{qb.x.ToFloat32(), qb.y.ToFloat32(), qb.z.ToFloat32(), qb.w.ToFloat32()};

    return (Common::Dot(a, b) < 0.f);
}

void Rasterizer::AddTriangle(const Pica::Shader::OutputVertex& v0,
                                   const Pica::Shader::OutputVertex& v1,
                                   const Pica::Shader::OutputVertex& v2) {
    vertex_batch.emplace_back(v0, false);
    vertex_batch.emplace_back(v1, AreQuaternionsOpposite(v0.quat, v1.quat));
    vertex_batch.emplace_back(v2, AreQuaternionsOpposite(v0.quat, v2.quat));
}

static constexpr std::array vs_attrib_types = {
    AttribType::Byte,  // VertexAttributeFormat::BYTE
    AttribType::Ubyte, // VertexAttributeFormat::UBYTE
    AttribType::Short, // VertexAttributeFormat::SHORT
    AttribType::Float  // VertexAttributeFormat::FLOAT
};

struct VertexArrayInfo {
    u32 vs_input_index_min;
    u32 vs_input_index_max;
    u32 vs_input_size;
};

Rasterizer::VertexArrayInfo Rasterizer::AnalyzeVertexArray(bool is_indexed) {
    const auto& regs = Pica::g_state.regs;
    const auto& vertex_attributes = regs.pipeline.vertex_attributes;

    u32 vertex_min;
    u32 vertex_max;
    if (is_indexed) {
        const auto& index_info = regs.pipeline.index_array;
        const PAddr address = vertex_attributes.GetPhysicalBaseAddress() + index_info.offset;
        const u8* index_address_8 = VideoCore::g_memory->GetPhysicalPointer(address);
        const u16* index_address_16 = reinterpret_cast<const u16*>(index_address_8);
        const bool index_u16 = index_info.format != 0;

        vertex_min = 0xFFFF;
        vertex_max = 0;
        const u32 size = regs.pipeline.num_vertices * (index_u16 ? 2 : 1);
        res_cache.FlushRegion(address, size, nullptr);
        for (u32 index = 0; index < regs.pipeline.num_vertices; ++index) {
            const u32 vertex = index_u16 ? index_address_16[index] : index_address_8[index];
            vertex_min = std::min(vertex_min, vertex);
            vertex_max = std::max(vertex_max, vertex);
        }
    } else {
        vertex_min = regs.pipeline.vertex_offset;
        vertex_max = regs.pipeline.vertex_offset + regs.pipeline.num_vertices - 1;
    }

    const u32 vertex_num = vertex_max - vertex_min + 1;
    u32 vs_input_size = 0;
    for (const auto& loader : vertex_attributes.attribute_loaders) {
        if (loader.component_count != 0) {
            vs_input_size += loader.byte_count * vertex_num;
        }
    }

    return {vertex_min, vertex_max, vs_input_size};
}

void Rasterizer::SetupVertexArray(u32 vs_input_size, u32 vs_input_index_min, u32 vs_input_index_max) {
    MICROPROFILE_SCOPE(VertexSetup);

    auto buffer_memory = vertex_buffer->Map(vs_input_size, 4);
    u8* array_ptr = buffer_memory.data();

    /**
     * The Nintendo 3DS has 12 attribute loaders which are used to tell the GPU
     * how to interpret vertex data. The program firsts sets GPUREG_ATTR_BUF_BASE to the base
     * address containing the vertex array data. The data for each attribute loader (i) can be found
     * by adding GPUREG_ATTR_BUFi_OFFSET to the base address. Attribute loaders can be thought
     * as something analogous to Vulkan bindings. The user can store attributes in separate loaders
     * or interleave them in the same loader.
     */
    const auto& regs = Pica::g_state.regs;
    const auto& vertex_attributes = regs.pipeline.vertex_attributes;
    PAddr base_address = vertex_attributes.GetPhysicalBaseAddress(); // GPUREG_ATTR_BUF_BASE

    VertexLayout layout{};
    std::array<bool, 16> enable_attributes{};
    std::array<u64, 16> binding_offsets{};

    u32 buffer_offset = 0;
    for (const auto& loader : vertex_attributes.attribute_loaders) {
        if (loader.component_count == 0 || loader.byte_count == 0) {
            continue;
        }

        // Analyze the attribute loader by checking which attributes it provides
        u32 offset = 0;
        for (u32 comp = 0; comp < loader.component_count && comp < 12; comp++) {
            u32 attribute_index = loader.GetComponent(comp);
            if (attribute_index < 12) {
                if (u32 size = vertex_attributes.GetNumElements(attribute_index); size != 0) {
                    offset = Common::AlignUp(offset, vertex_attributes.GetElementSizeInBytes(attribute_index));

                    const u32 input_reg = regs.vs.GetRegisterForAttribute(attribute_index);
                    const u32 attrib_format = static_cast<u32>(vertex_attributes.GetFormat(attribute_index));
                    const AttribType type = vs_attrib_types[attrib_format];

                    // Define the attribute
                    VertexAttribute& attribute = layout.attributes.at(layout.attribute_count++);
                    attribute.binding.Assign(layout.binding_count);
                    attribute.location.Assign(input_reg);
                    attribute.offset.Assign(offset);
                    attribute.type.Assign(type);
                    attribute.size.Assign(size);

                    enable_attributes[input_reg] = true;
                    offset += vertex_attributes.GetStride(attribute_index);
                }

            } else {
                // Attribute ids 12, 13, 14 and 15 signify 4, 8, 12 and 16-byte paddings respectively
                offset = Common::AlignUp(offset, 4);
                offset += (attribute_index - 11) * 4;
            }
        }

        PAddr data_addr = base_address + loader.data_offset + (vs_input_index_min * loader.byte_count);
        const u32 vertex_num = vs_input_index_max - vs_input_index_min + 1;
        const u32 data_size = loader.byte_count * vertex_num;

        res_cache.FlushRegion(data_addr, data_size, nullptr);
        std::memcpy(array_ptr, VideoCore::g_memory->GetPhysicalPointer(data_addr), data_size);

        // Create the binding associated with this loader
        VertexBinding& binding = layout.bindings.at(layout.binding_count);
        binding.binding.Assign(layout.binding_count);
        binding.fixed.Assign(0);
        binding.stride.Assign(loader.byte_count);

        // Keep track of the binding offsets so we can bind the vertex buffer later
        binding_offsets[layout.binding_count++] = buffer_offset;
        array_ptr += data_size;
        buffer_offset += data_size;
    }

    // Reserve the last binding for fixed attributes
    u32 offset = 0;
    for (std::size_t i = 0; i < 16; i++) {
        if (vertex_attributes.IsDefaultAttribute(i)) {
            const u32 reg = regs.vs.GetRegisterForAttribute(i);
            if (!enable_attributes[reg]) {
                const auto& attr = Pica::g_state.input_default_attributes.attr[i];
                const std::array data = {
                    attr.x.ToFloat32(),
                    attr.y.ToFloat32(),
                    attr.z.ToFloat32(),
                    attr.w.ToFloat32()
                };

                // Copy the data to the end of the buffer
                const u32 data_size = sizeof(float) * data.size();
                std::memcpy(array_ptr, data.data(), data_size);

                // Define the binding. Note that the counter is not incremented
                VertexBinding& binding = layout.bindings.at(layout.binding_count);
                binding.binding.Assign(layout.binding_count);
                binding.fixed.Assign(1);
                binding.stride.Assign(offset);

                VertexAttribute& attribute = layout.attributes.at(layout.attribute_count++);
                attribute.binding.Assign(layout.binding_count);
                attribute.location.Assign(reg);
                attribute.offset.Assign(offset);
                attribute.type.Assign(AttribType::Float);
                attribute.size.Assign(4);

                offset += data_size;
                array_ptr += data_size;
                binding_offsets[layout.binding_count] = buffer_offset;
            }
        }
    }

    // Upload data to the GPU
    vertex_buffer->Commit(vs_input_size);

    // Bind the vertex buffers with all the bindings
    auto offsets = std::span<u64>{binding_offsets.data(), layout.binding_count};
    backend->BindVertexBuffer(vertex_buffer, offsets);
}

bool Rasterizer::AccelerateDrawBatch(bool is_indexed) {
    const auto& regs = Pica::g_state.regs;
    if (regs.pipeline.use_gs != Pica::PipelineRegs::UseGS::No) {
        if (regs.pipeline.gs_config.mode != Pica::PipelineRegs::GSMode::Point) {
            return false;
        }

        if (regs.pipeline.triangle_topology != Pica::TriangleTopology::Shader) {
            return false;
        }

        LOG_ERROR(Render_Vulkan, "Accelerate draw doesn't support geometry shader");
        return false;
    }

    // Setup vertex shader
    MICROPROFILE_SCOPE(VertexShader);
    if (!pipeline_cache->UsePicaVertexShader(regs, Pica::g_state.vs)) {
        return false;
    }

    // Setup geometry shader
    MICROPROFILE_SCOPE(GeometryShader);
    pipeline_cache->UseFixedGeometryShader(regs);

    return Draw(true, is_indexed);
}

bool Rasterizer::AccelerateDrawBatchInternal(PipelineHandle pipeline, FramebufferHandle framebuffer, bool is_indexed) {
    const auto& regs = Pica::g_state.regs;

    auto [vs_input_index_min, vs_input_index_max, vs_input_size] = AnalyzeVertexArray(is_indexed);

    if (vs_input_size > VERTEX_BUFFER_INFO.capacity) {
        LOG_WARNING(Render_Vulkan, "Too large vertex input size {}", vs_input_size);
        return false;
    }

    SetupVertexArray(vs_input_size, vs_input_index_min, vs_input_index_max);

    if (is_indexed) {
        bool index_u16 = regs.pipeline.index_array.format != 0;
        const u64 index_buffer_size = regs.pipeline.num_vertices * (index_u16 ? 2 : 1);

        if (index_buffer_size > INDEX_BUFFER_INFO.capacity) {
            LOG_WARNING(Render_OpenGL, "Too large index input size {}", index_buffer_size);
            return false;
        }

        const u8* index_data = VideoCore::g_memory->GetPhysicalPointer(
            regs.pipeline.vertex_attributes.GetPhysicalBaseAddress() +
            regs.pipeline.index_array.offset);

        // Upload index buffer data to the GPU
        const u32 mapped_offset = index_buffer->GetCurrentOffset();
        auto buffer = index_buffer->Map(index_buffer_size, 4);
        std::memcpy(buffer.data(), index_data, index_buffer_size);
        index_buffer->Commit(index_buffer_size);

        backend->BindIndexBuffer(index_buffer, index_u16 ? AttribType::Short : AttribType::Ubyte, mapped_offset);
        backend->DrawIndexed(pipeline, framebuffer, vs_input_index_min, 0, regs.pipeline.num_vertices);
    } else {
        backend->Draw(pipeline, framebuffer, 0, regs.pipeline.num_vertices);
    }

    return true;
}

void Rasterizer::DrawTriangles() {
    if (vertex_batch.empty())
        return;
    Draw(false, false);
}

bool Rasterizer::Draw(bool accelerate, bool is_indexed) {
    MICROPROFILE_SCOPE(Drawing);
    const auto& regs = Pica::g_state.regs;

    bool shadow_rendering = regs.framebuffer.output_merger.fragment_operation_mode ==
                            Pica::FragmentOperationMode::Shadow;

    // Query framebuffer usage
    const bool has_stencil =
        regs.framebuffer.framebuffer.depth_format == Pica::FramebufferRegs::DepthFormat::D24S8;

    const bool write_color_fb =
        shadow_rendering || raster_info.blending.color_write_mask.Value() != 0;
    const bool write_depth_fb =
        (raster_info.depth_stencil.depth_test_enable && raster_info.depth_stencil.depth_write_enable) ||
        (has_stencil && raster_info.depth_stencil.stencil_test_enable && raster_info.depth_stencil.stencil_write_mask != 0);

    const bool using_color_fb =
        regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress() != 0 && write_color_fb;
    const bool using_depth_fb =
        !shadow_rendering && regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress() != 0 &&
        (write_depth_fb || regs.framebuffer.output_merger.depth_test_enable != 0 ||
         (has_stencil && raster_info.depth_stencil.stencil_test_enable));

    Common::Rectangle<s32> viewport_rect_unscaled{
        // These registers hold half-width and half-height, so must be multiplied by 2
        regs.rasterizer.viewport_corner.x,  // left
        regs.rasterizer.viewport_corner.y + // top
            static_cast<s32>(Pica::float24::FromRaw(regs.rasterizer.viewport_size_y).ToFloat32() * 2),
        regs.rasterizer.viewport_corner.x + // right
            static_cast<s32>(Pica::float24::FromRaw(regs.rasterizer.viewport_size_x).ToFloat32() * 2),
        regs.rasterizer.viewport_corner.y // bottom
    };

    // Retrive the render target surfaces from the cache
    Surface color_surface;
    Surface depth_surface;
    Common::Rectangle<u32> surfaces_rect;
    std::tie(color_surface, depth_surface, surfaces_rect) =
            res_cache.GetFramebufferSurfaces(using_color_fb, using_depth_fb, viewport_rect_unscaled);

    // Calucate the scaled viewport rectangle
    const u16 res_scale = color_surface != nullptr ? color_surface->res_scale
                                                   : (depth_surface == nullptr ? 1u : depth_surface->res_scale);

    Common::Rectangle<u32> draw_rect{
        static_cast<u32>(std::clamp<s32>(static_cast<s32>(surfaces_rect.left) +
                                             viewport_rect_unscaled.left * res_scale,
                                         surfaces_rect.left, surfaces_rect.right)), // Left
        static_cast<u32>(std::clamp<s32>(static_cast<s32>(surfaces_rect.bottom) +
                                             viewport_rect_unscaled.top * res_scale,
                                         surfaces_rect.bottom, surfaces_rect.top)), // Top
        static_cast<u32>(std::clamp<s32>(static_cast<s32>(surfaces_rect.left) +
                                             viewport_rect_unscaled.right * res_scale,
                                         surfaces_rect.left, surfaces_rect.right)), // Right
        static_cast<u32>(std::clamp<s32>(static_cast<s32>(surfaces_rect.bottom) +
                                             viewport_rect_unscaled.bottom * res_scale,
                                         surfaces_rect.bottom, surfaces_rect.top))}; // Bottom

    // Retrieve the framebuffer assigned to the surfaces and update raster_info
    FramebufferHandle framebuffer = res_cache.GetFramebuffer(color_surface, depth_surface);
    framebuffer->SetDrawRect(draw_rect);
    framebuffer->SetLoadOp(LoadOp::Load);

    raster_info.color_attachment = framebuffer->GetColorAttachment().IsValid() ?
                                   framebuffer->GetColorAttachment()->GetFormat() :
                                   TextureFormat::Undefined;
    raster_info.depth_attachment = framebuffer->GetDepthStencilAttachment().IsValid() ?
                                   framebuffer->GetDepthStencilAttachment()->GetFormat() :
                                   TextureFormat::Undefined;

    if (uniform_block_data.data.framebuffer_scale != res_scale) {
        uniform_block_data.data.framebuffer_scale = res_scale;
        uniform_block_data.dirty = true;
    }

    // Scissor checks are window-, not viewport-relative, which means that if the cached texture
    // sub-rect changes, the scissor bounds also need to be updated.
    int scissor_x1 = static_cast<int>(surfaces_rect.left + regs.rasterizer.scissor_test.x1 * res_scale);
    int scissor_y1 = static_cast<int>(surfaces_rect.bottom + regs.rasterizer.scissor_test.y1 * res_scale);

    // x2, y2 have +1 added to cover the entire pixel area, otherwise you might get cracks when
    // scaling or doing multisampling.
    int scissor_x2 = static_cast<int>(surfaces_rect.left + (regs.rasterizer.scissor_test.x2 + 1) * res_scale);
    int scissor_y2 = static_cast<int>(surfaces_rect.bottom + (regs.rasterizer.scissor_test.y2 + 1) * res_scale);

    if (uniform_block_data.data.scissor_x1 != scissor_x1 ||
        uniform_block_data.data.scissor_x2 != scissor_x2 ||
        uniform_block_data.data.scissor_y1 != scissor_y1 ||
        uniform_block_data.data.scissor_y2 != scissor_y2) {

        uniform_block_data.data.scissor_x1 = scissor_x1;
        uniform_block_data.data.scissor_x2 = scissor_x2;
        uniform_block_data.data.scissor_y1 = scissor_y1;
        uniform_block_data.data.scissor_y2 = scissor_y2;
        uniform_block_data.dirty = true;
    }

    // Bind shaders and retrieve rasterizer pipeline
    if (!accelerate) {
        pipeline_cache->UseTrivialVertexShader();
        pipeline_cache->UseTrivialGeometryShader();
    }

    // Sync and bind the shader
    if (shader_dirty) {
        pipeline_cache->UseFragmentShader(regs);
        shader_dirty = false;
    }

    // Sync the viewport
    PipelineHandle raster_pipeline = pipeline_cache->GetPipeline(raster_info);
    raster_pipeline->ApplyDynamic(raster_info);
    raster_pipeline->SetViewport(surfaces_rect.left + viewport_rect_unscaled.left * res_scale,
                                 surfaces_rect.bottom + viewport_rect_unscaled.bottom * res_scale,
                                 viewport_rect_unscaled.GetWidth() * res_scale,
                                 viewport_rect_unscaled.GetHeight() * res_scale);

    // Bind texel buffers
    raster_pipeline->BindBuffer(0, 2, texel_buffer_lut_lf);
    raster_pipeline->BindBuffer(0, 3, texel_buffer_lut);
    raster_pipeline->BindBuffer(0, 4, texel_buffer_lut, 0, WHOLE_SIZE, 1);

    // Checks if the game is trying to use a surface as a texture and framebuffer at the same time
    // which causes unpredictable behavior on the host.
    // Making a copy to sample from eliminates this issue and seems to be fairly cheap.
    TextureHandle temp_tex;
    auto CheckBarrier = [&](TextureHandle texture, u32 texture_index) {
        if (color_surface && color_surface->texture == texture) {
            temp_tex = backend->CreateTexture(texture->GetInfo());
            temp_tex->CopyFrom(texture);
            raster_pipeline->BindTexture(TEXTURE_GROUP, texture_index, temp_tex);
        } else {
            raster_pipeline->BindTexture(TEXTURE_GROUP, texture_index, texture);
        }
    };

    // Sync and bind the texture surfaces
    const auto pica_textures = regs.texturing.GetTextures();
    for (unsigned texture_index = 0; texture_index < pica_textures.size(); ++texture_index) {
        const auto& texture = pica_textures[texture_index];

        if (texture.enabled) {
            /*if (texture_index == 0) {
                using TextureType = Pica::TexturingRegs::TextureConfig::TextureType;
                switch (texture.config.type.Value()) {
                case TextureType::Shadow2D: {
                    if (!allow_shadow)
                        continue;

                    Surface surface = res_cache.GetTextureSurface(texture);
                    if (surface != nullptr) {
                        CheckBarrier(state.image_shadow_texture_px = surface->texture.handle);
                    } else {
                        state.image_shadow_texture_px = 0;
                    }
                    continue;
                }
                case TextureType::ShadowCube: {
                    if (!allow_shadow)
                        continue;
                    Pica::Texture::TextureInfo info = Pica::Texture::TextureInfo::FromPicaRegister(
                        texture.config, texture.format);
                    Surface surface;

                    using CubeFace = Pica::TexturingRegs::CubeFace;
                    info.physical_address =
                        regs.texturing.GetCubePhysicalAddress(CubeFace::PositiveX);
                    surface = res_cache.GetTextureSurface(info);
                    if (surface != nullptr) {
                        CheckBarrier(state.image_shadow_texture_px = surface->texture.handle);
                    } else {
                        state.image_shadow_texture_px = 0;
                    }

                    info.physical_address =
                        regs.texturing.GetCubePhysicalAddress(CubeFace::NegativeX);
                    surface = res_cache.GetTextureSurface(info);
                    if (surface != nullptr) {
                        CheckBarrier(state.image_shadow_texture_nx = surface->texture.handle);
                    } else {
                        state.image_shadow_texture_nx = 0;
                    }

                    info.physical_address =
                        regs.texturing.GetCubePhysicalAddress(CubeFace::PositiveY);
                    surface = res_cache.GetTextureSurface(info);
                    if (surface != nullptr) {
                        CheckBarrier(state.image_shadow_texture_py = surface->texture.handle);
                    } else {
                        state.image_shadow_texture_py = 0;
                    }

                    info.physical_address =
                        regs.texturing.GetCubePhysicalAddress(CubeFace::NegativeY);
                    surface = res_cache.GetTextureSurface(info);
                    if (surface != nullptr) {
                        CheckBarrier(state.image_shadow_texture_ny = surface->texture.handle);
                    } else {
                        state.image_shadow_texture_ny = 0;
                    }

                    info.physical_address =
                        regs.texturing.GetCubePhysicalAddress(CubeFace::PositiveZ);
                    surface = res_cache.GetTextureSurface(info);
                    if (surface != nullptr) {
                        CheckBarrier(state.image_shadow_texture_pz = surface->texture.handle);
                    } else {
                        state.image_shadow_texture_pz = 0;
                    }

                    info.physical_address =
                        regs.texturing.GetCubePhysicalAddress(CubeFace::NegativeZ);
                    surface = res_cache.GetTextureSurface(info);
                    if (surface != nullptr) {
                        CheckBarrier(state.image_shadow_texture_nz = surface->texture.handle);
                    } else {
                        state.image_shadow_texture_nz = 0;
                    }

                    continue;
                }
                case TextureType::TextureCube:
                    using CubeFace = Pica::TexturingRegs::CubeFace;
                    TextureCubeConfig config;
                    config.px = regs.texturing.GetCubePhysicalAddress(CubeFace::PositiveX);
                    config.nx = regs.texturing.GetCubePhysicalAddress(CubeFace::NegativeX);
                    config.py = regs.texturing.GetCubePhysicalAddress(CubeFace::PositiveY);
                    config.ny = regs.texturing.GetCubePhysicalAddress(CubeFace::NegativeY);
                    config.pz = regs.texturing.GetCubePhysicalAddress(CubeFace::PositiveZ);
                    config.nz = regs.texturing.GetCubePhysicalAddress(CubeFace::NegativeZ);
                    config.width = texture.config.width;
                    config.format = texture.format;
                    state.texture_cube_unit.texture_cube =
                        res_cache.GetTextureCube(config).texture.handle;

                    texture_cube_sampler.SyncWithConfig(texture.config);
                    state.texture_units[texture_index].texture_2d = 0;
                    continue; // Texture unit 0 setup finished. Continue to next unit
                default:
                    state.texture_cube_unit.texture_cube = 0;
                }
            }*/

            //texture_samplers[texture_index].SyncWithConfig(texture.config);

            // Update sampler key
            texture_samplers[texture_index] = SamplerInfo{
                .mag_filter = texture.config.mag_filter,
                .min_filter = texture.config.min_filter,
                .mip_filter = texture.config.mip_filter,
                .wrap_s = texture.config.wrap_s,
                .wrap_t = texture.config.wrap_t,
                .border_color = texture.config.border_color.raw,
                .lod_min = texture.config.lod.min_level,
                .lod_max = texture.config.lod.max_level,
                .lod_bias = texture.config.lod.bias
            };

            // Search the cache and bind the appropriate sampler
            const SamplerInfo& key = texture_samplers[texture_index];
            if (auto iter = sampler_cache.find(key); iter != sampler_cache.end()) {
                raster_pipeline->BindSampler(SAMPLER_GROUP, texture_index, iter->second);
            } else {
                SamplerHandle texture_sampler = backend->CreateSampler(key);
                sampler_cache.emplace(key, texture_sampler);
                raster_pipeline->BindSampler(SAMPLER_GROUP, texture_index, texture_sampler);
            }

            Surface surface = res_cache.GetTextureSurface(texture);
            if (surface != nullptr) {
                CheckBarrier(surface->texture, texture_index);
            } else {
                // Can occur when texture addr is null or its memory is unmapped/invalid
                // HACK: In this case, the correct behaviour for the PICA is to use the last
                // rendered colour. But because this would be impractical to implement, the
                // next best alternative is to use a clear texture, essentially skipping
                // the geometry in question.
                // For example: a bug in Pokemon X/Y causes NULL-texture squares to be drawn
                // on the male character's face, which in the OpenGL default appear black.
                //state.texture_units[texture_index].texture_2d = default_texture;
                raster_pipeline->BindTexture(TEXTURE_GROUP, texture_index, clear_texture);
            }
        } else {
            //state.texture_units[texture_index].texture_2d = 0;
            raster_pipeline->BindTexture(TEXTURE_GROUP, texture_index, clear_texture);
            raster_pipeline->BindSampler(SAMPLER_GROUP, texture_index, texture_cube_sampler);
        }
    }

    // TODO: Implement texture cubes
    raster_pipeline->BindTexture(TEXTURE_GROUP, 3, clear_texture);

    // TODO: Implement texture cubes
    raster_pipeline->BindSampler(SAMPLER_GROUP, 3, texture_cube_sampler);

    // Sync the LUTs within the texture buffer
    SyncAndUploadLUTs();
    SyncAndUploadLUTsLF();

    // Sync the uniform data
    UploadUniforms(raster_pipeline, accelerate);

    // Viewport can have negative offsets or larger dimensions than our framebuffer sub-rect.
    // Enable scissor test to prevent drawing outside of the framebuffer region
    raster_pipeline->SetScissor(draw_rect.left, draw_rect.bottom, draw_rect.GetWidth(), draw_rect.GetHeight());

    // Draw the vertex batch
    bool succeeded = true;
    if (accelerate) {
        succeeded = AccelerateDrawBatchInternal(raster_pipeline, framebuffer, is_indexed);
    } else {
        // Bind the vertex buffer at the current mapped offset. This effectively means
        // that when base_vertex is zero the GPU will start drawing from the current mapped
        // offset not the start of the buffer.
        const std::array<u64, 1> mapped_offset = {vertex_buffer->GetCurrentOffset()};
        backend->BindVertexBuffer(vertex_buffer, mapped_offset);

        const std::size_t max_vertices = VERTEX_BUFFER_INFO.capacity / sizeof(HardwareVertex);
        for (std::size_t base_vertex = 0; base_vertex < vertex_batch.size(); base_vertex += max_vertices) {
            const u32 vertices = std::min(max_vertices, vertex_batch.size() - base_vertex);
            const u32 vertex_size = vertices * sizeof(HardwareVertex);

            // Copy vertex data
            auto vertex_memory = vertex_buffer->Map(vertex_size, sizeof(HardwareVertex));
            std::memcpy(vertex_memory.data(), vertex_batch.data() + base_vertex, vertex_size);
            vertex_buffer->Commit(vertex_size);

            // Draw
            backend->Draw(raster_pipeline, framebuffer, base_vertex, vertices);
        }
    }

    vertex_batch.clear();

    // Mark framebuffer surfaces as dirty
    Common::Rectangle<u32> draw_rect_unscaled{
        draw_rect.left / res_scale,
        draw_rect.top / res_scale,
        draw_rect.right / res_scale,
        draw_rect.bottom / res_scale
    };

    if (color_surface != nullptr && write_color_fb) {
        auto interval = color_surface->GetSubRectInterval(draw_rect_unscaled);
        res_cache.InvalidateRegion(boost::icl::first(interval), boost::icl::length(interval),
                                   color_surface);
    }

    if (depth_surface != nullptr && write_depth_fb) {
        auto interval = depth_surface->GetSubRectInterval(draw_rect_unscaled);
        res_cache.InvalidateRegion(boost::icl::first(interval), boost::icl::length(interval),
                                   depth_surface);
    }

    return succeeded;
}

void Rasterizer::NotifyPicaRegisterChanged(u32 id) {
    const auto& regs = Pica::g_state.regs;

    switch (id) {
    // Culling
    case PICA_REG_INDEX(rasterizer.cull_mode):
        SyncCullMode();
        break;

    // Clipping plane
    case PICA_REG_INDEX(rasterizer.clip_enable):
        SyncClipEnabled();
        break;

    case PICA_REG_INDEX(rasterizer.clip_coef[0]):
    case PICA_REG_INDEX(rasterizer.clip_coef[1]):
    case PICA_REG_INDEX(rasterizer.clip_coef[2]):
    case PICA_REG_INDEX(rasterizer.clip_coef[3]):
        SyncClipCoef();
        break;

    // Depth modifiers
    case PICA_REG_INDEX(rasterizer.viewport_depth_range):
        SyncDepthScale();
        break;
    case PICA_REG_INDEX(rasterizer.viewport_depth_near_plane):
        SyncDepthOffset();
        break;

    // Depth buffering
    case PICA_REG_INDEX(rasterizer.depthmap_enable):
        shader_dirty = true;
        break;

    // Blending
    case PICA_REG_INDEX(framebuffer.output_merger.alphablend_enable):
        SyncBlendEnabled();
        break;
    case PICA_REG_INDEX(framebuffer.output_merger.alpha_blending):
        SyncBlendFuncs();
        break;
    case PICA_REG_INDEX(framebuffer.output_merger.blend_const):
        SyncBlendColor();
        break;

    // Shadow texture
    case PICA_REG_INDEX(texturing.shadow):
        SyncShadowTextureBias();
        break;

    // Fog state
    case PICA_REG_INDEX(texturing.fog_color):
        SyncFogColor();
        break;
    case PICA_REG_INDEX(texturing.fog_lut_data[0]):
    case PICA_REG_INDEX(texturing.fog_lut_data[1]):
    case PICA_REG_INDEX(texturing.fog_lut_data[2]):
    case PICA_REG_INDEX(texturing.fog_lut_data[3]):
    case PICA_REG_INDEX(texturing.fog_lut_data[4]):
    case PICA_REG_INDEX(texturing.fog_lut_data[5]):
    case PICA_REG_INDEX(texturing.fog_lut_data[6]):
    case PICA_REG_INDEX(texturing.fog_lut_data[7]):
        uniform_block_data.fog_lut_dirty = true;
        break;

    // ProcTex state
    case PICA_REG_INDEX(texturing.proctex):
    case PICA_REG_INDEX(texturing.proctex_lut):
    case PICA_REG_INDEX(texturing.proctex_lut_offset):
        SyncProcTexBias();
        shader_dirty = true;
        break;

    case PICA_REG_INDEX(texturing.proctex_noise_u):
    case PICA_REG_INDEX(texturing.proctex_noise_v):
    case PICA_REG_INDEX(texturing.proctex_noise_frequency):
        SyncProcTexNoise();
        break;

    case PICA_REG_INDEX(texturing.proctex_lut_data[0]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[1]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[2]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[3]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[4]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[5]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[6]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[7]):
        using Pica::TexturingRegs;
        switch (regs.texturing.proctex_lut_config.ref_table.Value()) {
        case TexturingRegs::ProcTexLutTable::Noise:
            uniform_block_data.proctex_noise_lut_dirty = true;
            break;
        case TexturingRegs::ProcTexLutTable::ColorMap:
            uniform_block_data.proctex_color_map_dirty = true;
            break;
        case TexturingRegs::ProcTexLutTable::AlphaMap:
            uniform_block_data.proctex_alpha_map_dirty = true;
            break;
        case TexturingRegs::ProcTexLutTable::Color:
            uniform_block_data.proctex_lut_dirty = true;
            break;
        case TexturingRegs::ProcTexLutTable::ColorDiff:
            uniform_block_data.proctex_diff_lut_dirty = true;
            break;
        }
        break;

    // Alpha test
    case PICA_REG_INDEX(framebuffer.output_merger.alpha_test):
        SyncAlphaTest();
        shader_dirty = true;
        break;

    // Sync GL stencil test + stencil write mask
    // (Pica stencil test function register also contains a stencil write mask)
    case PICA_REG_INDEX(framebuffer.output_merger.stencil_test.raw_func):
        SyncStencilTest();
        SyncStencilWriteMask();
        break;
    //case PICA_REG_INDEX(framebuffer.framebuffer.color_format):
        //raster_info.color_attachment =
    case PICA_REG_INDEX(framebuffer.output_merger.stencil_test.raw_op):
    case PICA_REG_INDEX(framebuffer.framebuffer.depth_format):
        SyncStencilTest();
        break;

    // Sync GL depth test + depth and color write mask
    // (Pica depth test function register also contains a depth and color write mask)
    case PICA_REG_INDEX(framebuffer.output_merger.depth_test_enable):
        SyncDepthTest();
        SyncDepthWriteMask();
        SyncColorWriteMask();
        break;

    // Sync GL depth and stencil write mask
    // (This is a dedicated combined depth / stencil write-enable register)
    case PICA_REG_INDEX(framebuffer.framebuffer.allow_depth_stencil_write):
        SyncDepthWriteMask();
        SyncStencilWriteMask();
        break;

    // Sync GL color write mask
    // (This is a dedicated color write-enable register)
    case PICA_REG_INDEX(framebuffer.framebuffer.allow_color_write):
        SyncColorWriteMask();
        break;

    case PICA_REG_INDEX(framebuffer.shadow):
        SyncShadowBias();
        break;

    // Scissor test
    case PICA_REG_INDEX(rasterizer.scissor_test.mode):
        shader_dirty = true;
        break;

    // Logic op
    case PICA_REG_INDEX(framebuffer.output_merger.logic_op):
        SyncLogicOp();
        break;

    case PICA_REG_INDEX(texturing.main_config):
        shader_dirty = true;
        break;

    // Texture 0 type
    case PICA_REG_INDEX(texturing.texture0.type):
        shader_dirty = true;
        break;

    // TEV stages
    // (This also syncs fog_mode and fog_flip which are part of tev_combiner_buffer_input)
    case PICA_REG_INDEX(texturing.tev_stage0.color_source1):
    case PICA_REG_INDEX(texturing.tev_stage0.color_modifier1):
    case PICA_REG_INDEX(texturing.tev_stage0.color_op):
    case PICA_REG_INDEX(texturing.tev_stage0.color_scale):
    case PICA_REG_INDEX(texturing.tev_stage1.color_source1):
    case PICA_REG_INDEX(texturing.tev_stage1.color_modifier1):
    case PICA_REG_INDEX(texturing.tev_stage1.color_op):
    case PICA_REG_INDEX(texturing.tev_stage1.color_scale):
    case PICA_REG_INDEX(texturing.tev_stage2.color_source1):
    case PICA_REG_INDEX(texturing.tev_stage2.color_modifier1):
    case PICA_REG_INDEX(texturing.tev_stage2.color_op):
    case PICA_REG_INDEX(texturing.tev_stage2.color_scale):
    case PICA_REG_INDEX(texturing.tev_stage3.color_source1):
    case PICA_REG_INDEX(texturing.tev_stage3.color_modifier1):
    case PICA_REG_INDEX(texturing.tev_stage3.color_op):
    case PICA_REG_INDEX(texturing.tev_stage3.color_scale):
    case PICA_REG_INDEX(texturing.tev_stage4.color_source1):
    case PICA_REG_INDEX(texturing.tev_stage4.color_modifier1):
    case PICA_REG_INDEX(texturing.tev_stage4.color_op):
    case PICA_REG_INDEX(texturing.tev_stage4.color_scale):
    case PICA_REG_INDEX(texturing.tev_stage5.color_source1):
    case PICA_REG_INDEX(texturing.tev_stage5.color_modifier1):
    case PICA_REG_INDEX(texturing.tev_stage5.color_op):
    case PICA_REG_INDEX(texturing.tev_stage5.color_scale):
    case PICA_REG_INDEX(texturing.tev_combiner_buffer_input):
        shader_dirty = true;
        break;
    case PICA_REG_INDEX(texturing.tev_stage0.const_r):
        SyncTevConstColor(0, regs.texturing.tev_stage0);
        break;
    case PICA_REG_INDEX(texturing.tev_stage1.const_r):
        SyncTevConstColor(1, regs.texturing.tev_stage1);
        break;
    case PICA_REG_INDEX(texturing.tev_stage2.const_r):
        SyncTevConstColor(2, regs.texturing.tev_stage2);
        break;
    case PICA_REG_INDEX(texturing.tev_stage3.const_r):
        SyncTevConstColor(3, regs.texturing.tev_stage3);
        break;
    case PICA_REG_INDEX(texturing.tev_stage4.const_r):
        SyncTevConstColor(4, regs.texturing.tev_stage4);
        break;
    case PICA_REG_INDEX(texturing.tev_stage5.const_r):
        SyncTevConstColor(5, regs.texturing.tev_stage5);
        break;

    // TEV combiner buffer color
    case PICA_REG_INDEX(texturing.tev_combiner_buffer_color):
        SyncCombinerColor();
        break;

    // Fragment lighting switches
    case PICA_REG_INDEX(lighting.disable):
    case PICA_REG_INDEX(lighting.max_light_index):
    case PICA_REG_INDEX(lighting.config0):
    case PICA_REG_INDEX(lighting.config1):
    case PICA_REG_INDEX(lighting.abs_lut_input):
    case PICA_REG_INDEX(lighting.lut_input):
    case PICA_REG_INDEX(lighting.lut_scale):
    case PICA_REG_INDEX(lighting.light_enable):
        break;

    // Fragment lighting specular 0 color
    case PICA_REG_INDEX(lighting.light[0].specular_0):
        SyncLightSpecular0(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].specular_0):
        SyncLightSpecular0(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].specular_0):
        SyncLightSpecular0(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].specular_0):
        SyncLightSpecular0(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].specular_0):
        SyncLightSpecular0(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].specular_0):
        SyncLightSpecular0(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].specular_0):
        SyncLightSpecular0(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].specular_0):
        SyncLightSpecular0(7);
        break;

    // Fragment lighting specular 1 color
    case PICA_REG_INDEX(lighting.light[0].specular_1):
        SyncLightSpecular1(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].specular_1):
        SyncLightSpecular1(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].specular_1):
        SyncLightSpecular1(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].specular_1):
        SyncLightSpecular1(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].specular_1):
        SyncLightSpecular1(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].specular_1):
        SyncLightSpecular1(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].specular_1):
        SyncLightSpecular1(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].specular_1):
        SyncLightSpecular1(7);
        break;

    // Fragment lighting diffuse color
    case PICA_REG_INDEX(lighting.light[0].diffuse):
        SyncLightDiffuse(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].diffuse):
        SyncLightDiffuse(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].diffuse):
        SyncLightDiffuse(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].diffuse):
        SyncLightDiffuse(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].diffuse):
        SyncLightDiffuse(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].diffuse):
        SyncLightDiffuse(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].diffuse):
        SyncLightDiffuse(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].diffuse):
        SyncLightDiffuse(7);
        break;

    // Fragment lighting ambient color
    case PICA_REG_INDEX(lighting.light[0].ambient):
        SyncLightAmbient(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].ambient):
        SyncLightAmbient(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].ambient):
        SyncLightAmbient(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].ambient):
        SyncLightAmbient(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].ambient):
        SyncLightAmbient(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].ambient):
        SyncLightAmbient(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].ambient):
        SyncLightAmbient(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].ambient):
        SyncLightAmbient(7);
        break;

    // Fragment lighting position
    case PICA_REG_INDEX(lighting.light[0].x):
    case PICA_REG_INDEX(lighting.light[0].z):
        SyncLightPosition(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].x):
    case PICA_REG_INDEX(lighting.light[1].z):
        SyncLightPosition(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].x):
    case PICA_REG_INDEX(lighting.light[2].z):
        SyncLightPosition(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].x):
    case PICA_REG_INDEX(lighting.light[3].z):
        SyncLightPosition(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].x):
    case PICA_REG_INDEX(lighting.light[4].z):
        SyncLightPosition(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].x):
    case PICA_REG_INDEX(lighting.light[5].z):
        SyncLightPosition(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].x):
    case PICA_REG_INDEX(lighting.light[6].z):
        SyncLightPosition(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].x):
    case PICA_REG_INDEX(lighting.light[7].z):
        SyncLightPosition(7);
        break;

    // Fragment spot lighting direction
    case PICA_REG_INDEX(lighting.light[0].spot_x):
    case PICA_REG_INDEX(lighting.light[0].spot_z):
        SyncLightSpotDirection(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].spot_x):
    case PICA_REG_INDEX(lighting.light[1].spot_z):
        SyncLightSpotDirection(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].spot_x):
    case PICA_REG_INDEX(lighting.light[2].spot_z):
        SyncLightSpotDirection(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].spot_x):
    case PICA_REG_INDEX(lighting.light[3].spot_z):
        SyncLightSpotDirection(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].spot_x):
    case PICA_REG_INDEX(lighting.light[4].spot_z):
        SyncLightSpotDirection(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].spot_x):
    case PICA_REG_INDEX(lighting.light[5].spot_z):
        SyncLightSpotDirection(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].spot_x):
    case PICA_REG_INDEX(lighting.light[6].spot_z):
        SyncLightSpotDirection(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].spot_x):
    case PICA_REG_INDEX(lighting.light[7].spot_z):
        SyncLightSpotDirection(7);
        break;

    // Fragment lighting light source config
    case PICA_REG_INDEX(lighting.light[0].config):
    case PICA_REG_INDEX(lighting.light[1].config):
    case PICA_REG_INDEX(lighting.light[2].config):
    case PICA_REG_INDEX(lighting.light[3].config):
    case PICA_REG_INDEX(lighting.light[4].config):
    case PICA_REG_INDEX(lighting.light[5].config):
    case PICA_REG_INDEX(lighting.light[6].config):
    case PICA_REG_INDEX(lighting.light[7].config):
        shader_dirty = true;
        break;

    // Fragment lighting distance attenuation bias
    case PICA_REG_INDEX(lighting.light[0].dist_atten_bias):
        SyncLightDistanceAttenuationBias(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].dist_atten_bias):
        SyncLightDistanceAttenuationBias(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].dist_atten_bias):
        SyncLightDistanceAttenuationBias(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].dist_atten_bias):
        SyncLightDistanceAttenuationBias(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].dist_atten_bias):
        SyncLightDistanceAttenuationBias(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].dist_atten_bias):
        SyncLightDistanceAttenuationBias(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].dist_atten_bias):
        SyncLightDistanceAttenuationBias(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].dist_atten_bias):
        SyncLightDistanceAttenuationBias(7);
        break;

    // Fragment lighting distance attenuation scale
    case PICA_REG_INDEX(lighting.light[0].dist_atten_scale):
        SyncLightDistanceAttenuationScale(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].dist_atten_scale):
        SyncLightDistanceAttenuationScale(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].dist_atten_scale):
        SyncLightDistanceAttenuationScale(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].dist_atten_scale):
        SyncLightDistanceAttenuationScale(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].dist_atten_scale):
        SyncLightDistanceAttenuationScale(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].dist_atten_scale):
        SyncLightDistanceAttenuationScale(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].dist_atten_scale):
        SyncLightDistanceAttenuationScale(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].dist_atten_scale):
        SyncLightDistanceAttenuationScale(7);
        break;

    // Fragment lighting global ambient color (emission + ambient * ambient)
    case PICA_REG_INDEX(lighting.global_ambient):
        SyncGlobalAmbient();
        break;

    // Fragment lighting lookup tables
    case PICA_REG_INDEX(lighting.lut_data[0]):
    case PICA_REG_INDEX(lighting.lut_data[1]):
    case PICA_REG_INDEX(lighting.lut_data[2]):
    case PICA_REG_INDEX(lighting.lut_data[3]):
    case PICA_REG_INDEX(lighting.lut_data[4]):
    case PICA_REG_INDEX(lighting.lut_data[5]):
    case PICA_REG_INDEX(lighting.lut_data[6]):
    case PICA_REG_INDEX(lighting.lut_data[7]): {
        const auto& lut_config = regs.lighting.lut_config;
        uniform_block_data.lighting_lut_dirty[lut_config.type] = true;
        uniform_block_data.lighting_lut_dirty_any = true;
        break;
    }
    }
}

void Rasterizer::FlushAll() {
    MICROPROFILE_SCOPE(CacheManagement);
    res_cache.FlushAll();
}

void Rasterizer::FlushRegion(PAddr addr, u32 size) {
    MICROPROFILE_SCOPE(CacheManagement);
    res_cache.FlushRegion(addr, size);
}

void Rasterizer::InvalidateRegion(PAddr addr, u32 size) {
    MICROPROFILE_SCOPE(CacheManagement);
    res_cache.InvalidateRegion(addr, size, nullptr);
}

void Rasterizer::FlushAndInvalidateRegion(PAddr addr, u32 size) {
    MICROPROFILE_SCOPE(CacheManagement);
    res_cache.FlushRegion(addr, size);
    res_cache.InvalidateRegion(addr, size, nullptr);
}

void Rasterizer::ClearAll(bool flush) {
    res_cache.ClearAll(flush);
}

bool Rasterizer::AccelerateDisplayTransfer(const GPU::Regs::DisplayTransferConfig& config) {
    MICROPROFILE_SCOPE(Blits);

    SurfaceParams src_params;
    src_params.addr = config.GetPhysicalInputAddress();
    src_params.width = config.output_width;
    src_params.stride = config.input_width;
    src_params.height = config.output_height;
    src_params.is_tiled = !config.input_linear;
    src_params.pixel_format = SurfaceParams::PixelFormatFromGPUPixelFormat(config.input_format);
    src_params.UpdateParams();

    SurfaceParams dst_params;
    dst_params.addr = config.GetPhysicalOutputAddress();
    dst_params.width = config.scaling != config.NoScale ? config.output_width.Value() / 2
                                                        : config.output_width.Value();
    dst_params.height = config.scaling == config.ScaleXY ? config.output_height.Value() / 2
                                                         : config.output_height.Value();
    dst_params.is_tiled = config.input_linear != config.dont_swizzle;
    dst_params.pixel_format = SurfaceParams::PixelFormatFromGPUPixelFormat(config.output_format);
    dst_params.UpdateParams();

    Common::Rectangle<u32> src_rect;
    Surface src_surface;
    std::tie(src_surface, src_rect) =
        res_cache.GetSurfaceSubRect(src_params, ScaleMatch::Ignore, true);
    if (src_surface == nullptr)
        return false;

    dst_params.res_scale = src_surface->res_scale;

    Common::Rectangle<u32> dst_rect;
    Surface dst_surface;
    std::tie(dst_surface, dst_rect) =
        res_cache.GetSurfaceSubRect(dst_params, ScaleMatch::Upscale, false);
    if (dst_surface == nullptr)
        return false;

    if (src_surface->is_tiled != dst_surface->is_tiled)
        std::swap(src_rect.top, src_rect.bottom);

    if (config.flip_vertically)
        std::swap(src_rect.top, src_rect.bottom);

    if (!res_cache.BlitSurfaces(src_surface, src_rect, dst_surface, dst_rect))
        return false;

    res_cache.InvalidateRegion(dst_params.addr, dst_params.size, dst_surface);
    return true;
}

bool Rasterizer::AccelerateTextureCopy(const GPU::Regs::DisplayTransferConfig& config) {
    u32 copy_size = Common::AlignDown(config.texture_copy.size, 16);
    if (copy_size == 0) {
        return false;
    }

    u32 input_gap = config.texture_copy.input_gap * 16;
    u32 input_width = config.texture_copy.input_width * 16;
    if (input_width == 0 && input_gap != 0) {
        return false;
    }
    if (input_gap == 0 || input_width >= copy_size) {
        input_width = copy_size;
        input_gap = 0;
    }
    if (copy_size % input_width != 0) {
        return false;
    }

    u32 output_gap = config.texture_copy.output_gap * 16;
    u32 output_width = config.texture_copy.output_width * 16;
    if (output_width == 0 && output_gap != 0) {
        return false;
    }
    if (output_gap == 0 || output_width >= copy_size) {
        output_width = copy_size;
        output_gap = 0;
    }
    if (copy_size % output_width != 0) {
        return false;
    }

    SurfaceParams src_params;
    src_params.addr = config.GetPhysicalInputAddress();
    src_params.stride = input_width + input_gap; // stride in bytes
    src_params.width = input_width;              // width in bytes
    src_params.height = copy_size / input_width;
    src_params.size = ((src_params.height - 1) * src_params.stride) + src_params.width;
    src_params.end = src_params.addr + src_params.size;

    Common::Rectangle<u32> src_rect;
    Surface src_surface;
    std::tie(src_surface, src_rect) = res_cache.GetTexCopySurface(src_params);
    if (src_surface == nullptr) {
        return false;
    }

    if (output_gap != 0 &&
        (output_width != src_surface->BytesInPixels(src_rect.GetWidth() / src_surface->res_scale) *
                             (src_surface->is_tiled ? 8 : 1) ||
         output_gap % src_surface->BytesInPixels(src_surface->is_tiled ? 64 : 1) != 0)) {
        return false;
    }

    SurfaceParams dst_params = *src_surface;
    dst_params.addr = config.GetPhysicalOutputAddress();
    dst_params.width = src_rect.GetWidth() / src_surface->res_scale;
    dst_params.stride = dst_params.width + src_surface->PixelsInBytes(
                                               src_surface->is_tiled ? output_gap / 8 : output_gap);
    dst_params.height = src_rect.GetHeight() / src_surface->res_scale;
    dst_params.res_scale = src_surface->res_scale;
    dst_params.UpdateParams();

    // Since we are going to invalidate the gap if there is one, we will have to load it first
    const bool load_gap = output_gap != 0;
    Common::Rectangle<u32> dst_rect;
    Surface dst_surface;
    std::tie(dst_surface, dst_rect) =
        res_cache.GetSurfaceSubRect(dst_params, ScaleMatch::Upscale, load_gap);
    if (dst_surface == nullptr) {
        return false;
    }

    if (dst_surface->type == SurfaceType::Texture) {
        return false;
    }

    if (!res_cache.BlitSurfaces(src_surface, src_rect, dst_surface, dst_rect)) {
        return false;
    }

    res_cache.InvalidateRegion(dst_params.addr, dst_params.size, dst_surface);
    return true;
}

bool Rasterizer::AccelerateFill(const GPU::Regs::MemoryFillConfig& config) {
    Surface dst_surface = res_cache.GetFillSurface(config);
    if (dst_surface == nullptr)
        return false;

    res_cache.InvalidateRegion(dst_surface->addr, dst_surface->size, dst_surface);
    return true;
}

bool Rasterizer::AccelerateDisplay(const GPU::Regs::FramebufferConfig& config,
                                         PAddr framebuffer_addr, u32 pixel_stride,
                                         ScreenInfo& screen_info) {
    if (framebuffer_addr == 0) {
        return false;
    }
    MICROPROFILE_SCOPE(CacheManagement);

    SurfaceParams src_params;
    src_params.addr = framebuffer_addr;
    src_params.width = std::min(config.width.Value(), pixel_stride);
    src_params.height = config.height;
    src_params.stride = pixel_stride;
    src_params.is_tiled = false;
    src_params.pixel_format = SurfaceParams::PixelFormatFromGPUPixelFormat(config.color_format);
    src_params.UpdateParams();

    Common::Rectangle<u32> src_rect;
    Surface src_surface;
    std::tie(src_surface, src_rect) =
        res_cache.GetSurfaceSubRect(src_params, ScaleMatch::Ignore, true);

    if (src_surface == nullptr) {
        return false;
    }

    u32 scaled_width = src_surface->GetScaledWidth();
    u32 scaled_height = src_surface->GetScaledHeight();

    screen_info.display_texcoords = Common::Rectangle<float>(
        (float)src_rect.bottom / (float)scaled_height, (float)src_rect.left / (float)scaled_width,
        (float)src_rect.top / (float)scaled_height, (float)src_rect.right / (float)scaled_width);

    screen_info.display_texture = src_surface->texture;

    return true;
}

void Rasterizer::SyncClipEnabled() {
    //state.clip_distance[1] = Pica::g_state.regs.rasterizer.clip_enable != 0;
}

void Rasterizer::SyncClipCoef() {
    const auto raw_clip_coef = Pica::g_state.regs.rasterizer.GetClipCoef();
    const Common::Vec4f new_clip_coef = {raw_clip_coef.x.ToFloat32(), raw_clip_coef.y.ToFloat32(),
                                         raw_clip_coef.z.ToFloat32(), raw_clip_coef.w.ToFloat32()};

    if (new_clip_coef != uniform_block_data.data.clip_coef) {
        uniform_block_data.data.clip_coef = new_clip_coef;
        uniform_block_data.dirty = true;
    }
}

void Rasterizer::SyncCullMode() {
    const auto& regs = Pica::g_state.regs;
    raster_info.rasterization.cull_mode.Assign(regs.rasterizer.cull_mode);
}

void Rasterizer::SyncDepthScale() {
    float depth_scale = Pica::float24::FromRaw(Pica::g_state.regs.rasterizer.viewport_depth_range).ToFloat32();

    if (depth_scale != uniform_block_data.data.depth_scale) {
        uniform_block_data.data.depth_scale = depth_scale;
        uniform_block_data.dirty = true;
    }
}

void Rasterizer::SyncDepthOffset() {
    float depth_offset = Pica::float24::FromRaw(Pica::g_state.regs.rasterizer.viewport_depth_near_plane).ToFloat32();

    if (depth_offset != uniform_block_data.data.depth_offset) {
        uniform_block_data.data.depth_offset = depth_offset;
        uniform_block_data.dirty = true;
    }
}

void Rasterizer::SyncBlendEnabled() {
    raster_info.blending.blend_enable.Assign(Pica::g_state.regs.framebuffer.output_merger.alphablend_enable);
}

void Rasterizer::SyncBlendFuncs() {
    const auto& regs = Pica::g_state.regs;

    raster_info.blending.color_blend_eq.Assign(regs.framebuffer.output_merger.alpha_blending.blend_equation_rgb);
    raster_info.blending.alpha_blend_eq.Assign(regs.framebuffer.output_merger.alpha_blending.blend_equation_a);
    raster_info.blending.src_color_blend_factor.Assign(regs.framebuffer.output_merger.alpha_blending.factor_source_rgb);
    raster_info.blending.dst_color_blend_factor.Assign(regs.framebuffer.output_merger.alpha_blending.factor_dest_rgb);
    raster_info.blending.src_alpha_blend_factor.Assign(regs.framebuffer.output_merger.alpha_blending.factor_source_a);
    raster_info.blending.dst_alpha_blend_factor.Assign(regs.framebuffer.output_merger.alpha_blending.factor_dest_a);
}

void Rasterizer::SyncBlendColor() {
    /*auto blend_color =
        PicaToGL::ColorRGBA8(Pica::g_state.regs.framebuffer.output_merger.blend_const.raw);
    state.blend.color.red = blend_color[0];
    state.blend.color.green = blend_color[1];
    state.blend.color.blue = blend_color[2];
    state.blend.color.alpha = blend_color[3];*/
}

void Rasterizer::SyncFogColor() {
    const auto& regs = Pica::g_state.regs;
    uniform_block_data.data.fog_color = {
        regs.texturing.fog_color.r.Value() / 255.0f,
        regs.texturing.fog_color.g.Value() / 255.0f,
        regs.texturing.fog_color.b.Value() / 255.0f,
    };
    uniform_block_data.dirty = true;
}

void Rasterizer::SyncProcTexNoise() {
    const auto& regs = Pica::g_state.regs.texturing;
    uniform_block_data.data.proctex_noise_f = {
        Pica::float16::FromRaw(regs.proctex_noise_frequency.u).ToFloat32(),
        Pica::float16::FromRaw(regs.proctex_noise_frequency.v).ToFloat32(),
    };
    uniform_block_data.data.proctex_noise_a = {
        regs.proctex_noise_u.amplitude / 4095.0f,
        regs.proctex_noise_v.amplitude / 4095.0f,
    };
    uniform_block_data.data.proctex_noise_p = {
        Pica::float16::FromRaw(regs.proctex_noise_u.phase).ToFloat32(),
        Pica::float16::FromRaw(regs.proctex_noise_v.phase).ToFloat32(),
    };

    uniform_block_data.dirty = true;
}

void Rasterizer::SyncProcTexBias() {
    const auto& regs = Pica::g_state.regs.texturing;
    uniform_block_data.data.proctex_bias =
        Pica::float16::FromRaw(regs.proctex.bias_low | (regs.proctex_lut.bias_high << 8))
            .ToFloat32();

    uniform_block_data.dirty = true;
}

void Rasterizer::SyncAlphaTest() {
    const auto& regs = Pica::g_state.regs;
    if (regs.framebuffer.output_merger.alpha_test.ref != uniform_block_data.data.alphatest_ref) {
        uniform_block_data.data.alphatest_ref = regs.framebuffer.output_merger.alpha_test.ref;
        uniform_block_data.dirty = true;
    }
}

void Rasterizer::SyncLogicOp() {
    const auto& regs = Pica::g_state.regs;
    raster_info.blending.logic_op.Assign(regs.framebuffer.output_merger.logic_op);
}

void Rasterizer::SyncColorWriteMask() {
    const auto& regs = Pica::g_state.regs;

    const u32 color_mask = (regs.framebuffer.output_merger.depth_color_mask >> 8) & 0xF;
    raster_info.blending.color_write_mask.Assign(color_mask);
}

void Rasterizer::SyncStencilWriteMask() {
    const auto& regs = Pica::g_state.regs;
    raster_info.depth_stencil.stencil_write_mask =
        (regs.framebuffer.framebuffer.allow_depth_stencil_write != 0)
            ? static_cast<u32>(regs.framebuffer.output_merger.stencil_test.write_mask)
            : 0;
}

void Rasterizer::SyncDepthWriteMask() {
    const auto& regs = Pica::g_state.regs;
    raster_info.depth_stencil.depth_write_enable.Assign(
                (regs.framebuffer.framebuffer.allow_depth_stencil_write != 0 &&
                 regs.framebuffer.output_merger.depth_write_enable));
}

void Rasterizer::SyncStencilTest() {
    const auto& regs = Pica::g_state.regs;

    raster_info.depth_stencil.stencil_test_enable.Assign(regs.framebuffer.output_merger.stencil_test.enable &&
                        regs.framebuffer.framebuffer.depth_format == Pica::FramebufferRegs::DepthFormat::D24S8);
    raster_info.depth_stencil.stencil_fail_op.Assign(regs.framebuffer.output_merger.stencil_test.action_stencil_fail);
    raster_info.depth_stencil.stencil_pass_op.Assign(regs.framebuffer.output_merger.stencil_test.action_depth_pass);
    raster_info.depth_stencil.stencil_depth_fail_op.Assign(regs.framebuffer.output_merger.stencil_test.action_depth_fail);
    raster_info.depth_stencil.stencil_compare_op.Assign(regs.framebuffer.output_merger.stencil_test.func);
    raster_info.depth_stencil.stencil_reference = regs.framebuffer.output_merger.stencil_test.reference_value;
    raster_info.depth_stencil.stencil_write_mask = regs.framebuffer.output_merger.stencil_test.input_mask;
}

void Rasterizer::SyncDepthTest() {
    const auto& regs = Pica::g_state.regs;
    raster_info.depth_stencil.depth_test_enable.Assign(regs.framebuffer.output_merger.depth_test_enable == 1 ||
                                    regs.framebuffer.output_merger.depth_write_enable == 1);
    raster_info.depth_stencil.depth_compare_op.Assign(
        regs.framebuffer.output_merger.depth_test_enable == 1
            ? regs.framebuffer.output_merger.depth_test_func.Value()
            : Pica::CompareFunc::Always);
}

void Rasterizer::SyncCombinerColor() {
    auto combiner_color = ColorRGBA8(Pica::g_state.regs.texturing.tev_combiner_buffer_color.raw);
    if (combiner_color != uniform_block_data.data.tev_combiner_buffer_color) {
        uniform_block_data.data.tev_combiner_buffer_color = combiner_color;
        uniform_block_data.dirty = true;
    }
}

void Rasterizer::SyncTevConstColor(std::size_t stage_index,
                                         const Pica::TexturingRegs::TevStageConfig& tev_stage) {
    const auto const_color = ColorRGBA8(tev_stage.const_color);

    if (const_color == uniform_block_data.data.const_color[stage_index]) {
        return;
    }

    uniform_block_data.data.const_color[stage_index] = const_color;
    uniform_block_data.dirty = true;
}

void Rasterizer::SyncGlobalAmbient() {
    auto color = LightColor(Pica::g_state.regs.lighting.global_ambient);
    if (color != uniform_block_data.data.lighting_global_ambient) {
        uniform_block_data.data.lighting_global_ambient = color;
        uniform_block_data.dirty = true;
    }
}

void Rasterizer::SyncLightSpecular0(int light_index) {
    auto color = LightColor(Pica::g_state.regs.lighting.light[light_index].specular_0);
    if (color != uniform_block_data.data.light_src[light_index].specular_0) {
        uniform_block_data.data.light_src[light_index].specular_0 = color;
        uniform_block_data.dirty = true;
    }
}

void Rasterizer::SyncLightSpecular1(int light_index) {
    auto color = LightColor(Pica::g_state.regs.lighting.light[light_index].specular_1);
    if (color != uniform_block_data.data.light_src[light_index].specular_1) {
        uniform_block_data.data.light_src[light_index].specular_1 = color;
        uniform_block_data.dirty = true;
    }
}

void Rasterizer::SyncLightDiffuse(int light_index) {
    auto color = LightColor(Pica::g_state.regs.lighting.light[light_index].diffuse);
    if (color != uniform_block_data.data.light_src[light_index].diffuse) {
        uniform_block_data.data.light_src[light_index].diffuse = color;
        uniform_block_data.dirty = true;
    }
}

void Rasterizer::SyncLightAmbient(int light_index) {
    auto color = LightColor(Pica::g_state.regs.lighting.light[light_index].ambient);
    if (color != uniform_block_data.data.light_src[light_index].ambient) {
        uniform_block_data.data.light_src[light_index].ambient = color;
        uniform_block_data.dirty = true;
    }
}

void Rasterizer::SyncLightPosition(int light_index) {
    const Common::Vec3f position = {
        Pica::float16::FromRaw(Pica::g_state.regs.lighting.light[light_index].x).ToFloat32(),
        Pica::float16::FromRaw(Pica::g_state.regs.lighting.light[light_index].y).ToFloat32(),
        Pica::float16::FromRaw(Pica::g_state.regs.lighting.light[light_index].z).ToFloat32()
    };

    if (position != uniform_block_data.data.light_src[light_index].position) {
        uniform_block_data.data.light_src[light_index].position = position;
        uniform_block_data.dirty = true;
    }
}

void Rasterizer::SyncLightSpotDirection(int light_index) {
    const auto& light = Pica::g_state.regs.lighting.light[light_index];
    const auto spot_direction = Common::Vec3f{light.spot_x, light.spot_y, light.spot_z} / 2047.0f;

    if (spot_direction != uniform_block_data.data.light_src[light_index].spot_direction) {
        uniform_block_data.data.light_src[light_index].spot_direction = spot_direction;
        uniform_block_data.dirty = true;
    }
}

void Rasterizer::SyncLightDistanceAttenuationBias(int light_index) {
    float dist_atten_bias = Pica::float20::FromRaw(Pica::g_state.regs.lighting.light[light_index].dist_atten_bias)
                                        .ToFloat32();

    if (dist_atten_bias != uniform_block_data.data.light_src[light_index].dist_atten_bias) {
        uniform_block_data.data.light_src[light_index].dist_atten_bias = dist_atten_bias;
        uniform_block_data.dirty = true;
    }
}

void Rasterizer::SyncLightDistanceAttenuationScale(int light_index) {
    float dist_atten_scale = Pica::float20::FromRaw(Pica::g_state.regs.lighting.light[light_index].dist_atten_scale)
                                         .ToFloat32();

    if (dist_atten_scale != uniform_block_data.data.light_src[light_index].dist_atten_scale) {
        uniform_block_data.data.light_src[light_index].dist_atten_scale = dist_atten_scale;
        uniform_block_data.dirty = true;
    }
}

void Rasterizer::SyncShadowBias() {
    const auto& shadow = Pica::g_state.regs.framebuffer.shadow;
    float constant = Pica::float16::FromRaw(shadow.constant).ToFloat32();
    float linear = Pica::float16::FromRaw(shadow.linear).ToFloat32();

    if (constant != uniform_block_data.data.shadow_bias_constant ||
        linear != uniform_block_data.data.shadow_bias_linear) {
        uniform_block_data.data.shadow_bias_constant = constant;
        uniform_block_data.data.shadow_bias_linear = linear;
        uniform_block_data.dirty = true;
    }
}

void Rasterizer::SyncShadowTextureBias() {
    int bias = Pica::g_state.regs.texturing.shadow.bias << 1;
    if (bias != uniform_block_data.data.shadow_texture_bias) {
        uniform_block_data.data.shadow_texture_bias = bias;
        uniform_block_data.dirty = true;
    }
}

void Rasterizer::SyncAndUploadLUTsLF() {
    constexpr std::size_t max_size = sizeof(Common::Vec2f) * 256 * Pica::LightingRegs::NumLightingSampler +
                                     sizeof(Common::Vec2f) * 128; // fog

    if (!uniform_block_data.lighting_lut_dirty_any && !uniform_block_data.fog_lut_dirty) {
        return;
    }

    std::size_t bytes_used = 0;
    auto buffer_ptr = texel_buffer_lut_lf->Map(max_size, sizeof(Common::Vec4f));
    const bool invalidate = texel_buffer_lut_lf->IsInvalid();
    const u32 offset = texel_buffer_lut_lf->GetCurrentOffset();

    // Sync the lighting luts
    if (uniform_block_data.lighting_lut_dirty_any || invalidate) {
        for (u32 index = 0; index < uniform_block_data.lighting_lut_dirty.size(); index++) {
            if (uniform_block_data.lighting_lut_dirty[index] || invalidate) {
                std::array<Common::Vec2f, 256> new_data;
                const auto& source_lut = Pica::g_state.lighting.luts[index];
                std::ranges::transform(source_lut, new_data.begin(), [](const auto& entry) {
                    return Common::Vec2f{entry.ToFloat(), entry.DiffToFloat()};
                });

                if (new_data != lighting_lut_data[index] || invalidate) {
                    lighting_lut_data[index] = new_data;
                    std::memcpy(buffer_ptr.data() + bytes_used, new_data.data(),
                                new_data.size() * sizeof(Common::Vec2f));
                    uniform_block_data.data.lighting_lut_offset[index / 4][index % 4] =
                        static_cast<int>((offset + bytes_used) / sizeof(Common::Vec2f));

                    uniform_block_data.dirty = true;
                    bytes_used += new_data.size() * sizeof(Common::Vec2f);
                }

                uniform_block_data.lighting_lut_dirty[index] = false;
            }
        }

        uniform_block_data.lighting_lut_dirty_any = false;
    }

    // Sync the fog lut
    if (uniform_block_data.fog_lut_dirty || invalidate) {
        std::array<Common::Vec2f, 128> new_data;

        std::ranges::transform(Pica::g_state.fog.lut, new_data.begin(), [](const auto& entry) {
            return Common::Vec2f{entry.ToFloat(), entry.DiffToFloat()};
        });

        if (new_data != fog_lut_data || invalidate) {
            fog_lut_data = new_data;
            std::memcpy(buffer_ptr.data() + bytes_used, new_data.data(), new_data.size() * sizeof(Common::Vec2f));
            uniform_block_data.data.fog_lut_offset =
                static_cast<int>((offset + bytes_used) / sizeof(Common::Vec2f));
            uniform_block_data.dirty = true;
            bytes_used += new_data.size() * sizeof(Common::Vec2f);
        }
        uniform_block_data.fog_lut_dirty = false;
    }

    if (bytes_used > 0) {
        texel_buffer_lut_lf->Commit(bytes_used);
    }
}

void Rasterizer::SyncAndUploadLUTs() {
    constexpr std::size_t max_size = sizeof(Common::Vec2f) * 128 * 3 + // proctex: noise + color + alpha
                                     sizeof(Common::Vec4f) * 256 +     // proctex
                                     sizeof(Common::Vec4f) * 256;      // proctex diff

    if (!uniform_block_data.proctex_noise_lut_dirty &&
        !uniform_block_data.proctex_color_map_dirty &&
        !uniform_block_data.proctex_alpha_map_dirty && !uniform_block_data.proctex_lut_dirty &&
        !uniform_block_data.proctex_diff_lut_dirty) {
        return;
    }

    std::size_t bytes_used = 0;
    auto buffer = texel_buffer_lut->Map(max_size, sizeof(Common::Vec4f));
    const bool invalidate = texel_buffer_lut->IsInvalid();
    const u32 offset = texel_buffer_lut->GetCurrentOffset();

    // helper function for SyncProcTexNoiseLUT/ColorMap/AlphaMap
    auto SyncProcTexValueLUT = [&](const std::array<Pica::State::ProcTex::ValueEntry, 128>& lut,
                                   std::array<Common::Vec2f, 128>& lut_data, int& lut_offset) {
        std::array<Common::Vec2f, 128> new_data;
        std::ranges::transform(lut, new_data.begin(), [](const auto& entry) {
            return Common::Vec2f{entry.ToFloat(), entry.DiffToFloat()};
        });

        if (new_data != lut_data || invalidate) {
            lut_data = new_data;
            std::memcpy(buffer.data() + bytes_used, new_data.data(), new_data.size() * sizeof(Common::Vec2f));

            lut_offset = static_cast<int>((offset + bytes_used) / sizeof(Common::Vec2f));
            uniform_block_data.dirty = true;
            bytes_used += new_data.size() * sizeof(Common::Vec2f);
        }
    };

    // Sync the proctex noise lut
    if (uniform_block_data.proctex_noise_lut_dirty || invalidate) {
        SyncProcTexValueLUT(Pica::g_state.proctex.noise_table, proctex_noise_lut_data,
                            uniform_block_data.data.proctex_noise_lut_offset);
        uniform_block_data.proctex_noise_lut_dirty = false;
    }

    // Sync the proctex color map
    if (uniform_block_data.proctex_color_map_dirty || invalidate) {
        SyncProcTexValueLUT(Pica::g_state.proctex.color_map_table, proctex_color_map_data,
                            uniform_block_data.data.proctex_color_map_offset);
        uniform_block_data.proctex_color_map_dirty = false;
    }

    // Sync the proctex alpha map
    if (uniform_block_data.proctex_alpha_map_dirty || invalidate) {
        SyncProcTexValueLUT(Pica::g_state.proctex.alpha_map_table, proctex_alpha_map_data,
                            uniform_block_data.data.proctex_alpha_map_offset);
        uniform_block_data.proctex_alpha_map_dirty = false;
    }

    // Sync the proctex lut
    if (uniform_block_data.proctex_lut_dirty || invalidate) {
        std::array<Common::Vec4f, 256> new_data;

        std::ranges::transform(Pica::g_state.proctex.color_table, new_data.begin(), [](const auto& entry) {
            auto rgba = entry.ToVector() / 255.0f;
            return Common::Vec4f{rgba.r(), rgba.g(), rgba.b(), rgba.a()};
        });

        if (new_data != proctex_lut_data || invalidate) {
            proctex_lut_data = new_data;
            std::memcpy(buffer.data() + bytes_used, new_data.data(), new_data.size() * sizeof(Common::Vec4f));
            uniform_block_data.data.proctex_lut_offset =
                static_cast<int>((offset + bytes_used) / sizeof(Common::Vec4f));
            uniform_block_data.dirty = true;
            bytes_used += new_data.size() * sizeof(Common::Vec4f);
        }

        uniform_block_data.proctex_lut_dirty = false;
    }

    // Sync the proctex difference lut
    if (uniform_block_data.proctex_diff_lut_dirty || invalidate) {
        std::array<Common::Vec4f, 256> new_data;

        std::ranges::transform(Pica::g_state.proctex.color_diff_table, new_data.begin(), [](const auto& entry) {
            auto rgba = entry.ToVector() / 255.0f;
            return Common::Vec4f{rgba.r(), rgba.g(), rgba.b(), rgba.a()};
        });

        if (new_data != proctex_diff_lut_data || invalidate) {
            proctex_diff_lut_data = new_data;
            std::memcpy(buffer.data() + bytes_used, new_data.data(), new_data.size() * sizeof(Common::Vec4f));
            uniform_block_data.data.proctex_diff_lut_offset =
                static_cast<int>((offset + bytes_used) / sizeof(Common::Vec4f));
            uniform_block_data.dirty = true;
            bytes_used += new_data.size() * sizeof(Common::Vec4f);
        }

        uniform_block_data.proctex_diff_lut_dirty = false;
    }

    if (bytes_used > 0) {
        texel_buffer_lut->Commit(bytes_used);
    }
}

void Rasterizer::UploadUniforms(PipelineHandle pipeline, bool accelerate_draw) {
    bool sync_vs = accelerate_draw;
    bool sync_fs = uniform_block_data.dirty;

    if (!sync_vs && !sync_fs) {
        return;
    }

    if (sync_vs) {
        VSUniformData vs_uniforms;
        vs_uniforms.uniforms.SetFromRegs(Pica::g_state.regs.vs, Pica::g_state.vs);

        auto uniforms = uniform_buffer_vs->Map(uniform_size_aligned_vs, uniform_buffer_alignment);
        uniform_block_data.current_vs_offset = uniform_buffer_vs->GetCurrentOffset();

        std::memcpy(uniforms.data(), &vs_uniforms, sizeof(vs_uniforms));
        uniform_buffer_vs->Commit(uniform_size_aligned_vs);
    }

    if (sync_fs) {
        auto uniforms = uniform_buffer_fs->Map(uniform_size_aligned_fs, uniform_buffer_alignment);
        uniform_block_data.current_fs_offset = uniform_buffer_fs->GetCurrentOffset();

        std::memcpy(uniforms.data(), &uniform_block_data.data, sizeof(UniformData));

        uniform_block_data.dirty = false;
        uniform_buffer_fs->Commit(uniform_size_aligned_fs);
    }

    // Bind updated ranges
    pipeline->BindBuffer(UTILITY_GROUP, 0, uniform_buffer_vs, uniform_block_data.current_vs_offset,
                         sizeof(VSUniformData));
    pipeline->BindBuffer(UTILITY_GROUP, 1, uniform_buffer_fs, uniform_block_data.current_fs_offset,
                         sizeof(UniformData));
}

} // namespace VideoCore
