// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/bit_field.h"
#include "common/bit_field_array.h"
#include "common/hash.h"
#include "video_core/common/buffer.h"
#include "video_core/common/texture.h"
#include "video_core/common/shader.h"
#include "video_core/regs_framebuffer.h"
#include "video_core/regs_rasterizer.h"
#include "video_core/regs_pipeline.h"

namespace VideoCore {

constexpr u32 MAX_SHADER_STAGES = 3;
constexpr u32 MAX_VERTEX_ATTRIBUTES = 16;
constexpr u32 MAX_VERTEX_BINDINGS = 16;
constexpr u32 MAX_BINDINGS_IN_GROUP = 7;
constexpr u32 MAX_BINDING_GROUPS = 6;

enum class PipelineType : u8 {
    Compute = 0,
    Graphics = 1
};

enum class BindingType : u32 {
    None = 0,
    Uniform = 1,
    UniformDynamic = 2,
    TexelBuffer = 3,
    Texture = 4,
    Sampler = 5,
    StorageImage = 6
};

using BindingGroup = BitFieldArray<0, 3, MAX_BINDINGS_IN_GROUP, BindingType>;

/**
 * Describes all the resources used in the pipeline
 */
struct PipelineLayoutInfo {
    u8 group_count = 0;
    std::array<BindingGroup, MAX_BINDING_GROUPS> binding_groups{};
    u8 push_constant_block_size = 0;
};

/**
 * The pipeline state is tightly packed with bitfields to reduce
 * the overhead of hashing as much as possible
 */
union RasterizationState {
    u8 value = 0;
    BitField<0, 2, Pica::TriangleTopology> topology;
    BitField<4, 2, Pica::CullMode> cull_mode;
};

struct DepthStencilState {
    union {
        u32 value = 0;
        BitField<0, 1, u32> depth_test_enable;
        BitField<1, 1, u32> depth_write_enable;
        BitField<2, 1, u32> stencil_test_enable;
        BitField<3, 3, Pica::CompareFunc> depth_compare_op;
        BitField<6, 3, Pica::StencilAction> stencil_fail_op;
        BitField<9, 3, Pica::StencilAction> stencil_pass_op;
        BitField<12, 3, Pica::StencilAction> stencil_depth_fail_op;
        BitField<15, 3, Pica::CompareFunc> stencil_compare_op;
    };

    // These are dynamic on most graphics APIs so keep them separate
    u8 stencil_reference;
    u8 stencil_compare_mask;
    u8 stencil_write_mask;
};

union BlendState {
    u32 value = 0;
    BitField<0, 1, u32> blend_enable;
    BitField<1, 4, Pica::BlendFactor> src_color_blend_factor;
    BitField<5, 4, Pica::BlendFactor> dst_color_blend_factor;
    BitField<9, 3, Pica::BlendEquation> color_blend_eq;
    BitField<12, 4, Pica::BlendFactor> src_alpha_blend_factor;
    BitField<16, 4, Pica::BlendFactor> dst_alpha_blend_factor;
    BitField<20, 3, Pica::BlendEquation> alpha_blend_eq;
    BitField<23, 4, u32> color_write_mask;
    BitField<27, 1, u32> logic_op_enable;
    BitField<28, 4, Pica::LogicOp> logic_op;
};

enum class AttribType : u32 {
    Float = 0,
    Int = 1,
    Short = 2,
    Byte = 3,
    Ubyte = 4
};

union VertexBinding {
    BitField<0, 4, u16> binding;
    BitField<4, 1, u16> fixed;
    BitField<5, 11, u16> stride;
};

union VertexAttribute {
    BitField<0, 4, u32> binding;
    BitField<4, 4, u32> location;
    BitField<8, 3, AttribType> type;
    BitField<11, 3, u32> size;
    BitField<14, 11, u32> offset;
};

#pragma pack(1)
struct VertexLayout {
    u8 binding_count = 0;
    u8 attribute_count = 0;
    std::array<VertexBinding, MAX_VERTEX_BINDINGS> bindings{};
    std::array<VertexAttribute, MAX_VERTEX_ATTRIBUTES> attributes{};
};
#pragma pack()

/**
 * Information about a graphics/compute pipeline
 */
#pragma pack(1)
struct PipelineInfo {
    std::array<ShaderHandle, MAX_SHADER_STAGES> shaders{};
    VertexLayout vertex_layout{};
    PipelineLayoutInfo layout{};
    BlendState blending{};
    TextureFormat color_attachment = TextureFormat::RGBA8;
    TextureFormat depth_attachment = TextureFormat::D24S8;
    RasterizationState rasterization{};
    DepthStencilState depth_stencil{};
};
#pragma pack()

constexpr s32 WHOLE_SIZE = -1;

// An opaque handle to a backend specific program pipeline
class PipelineBase : public IntrusivePtrEnabled<PipelineBase> {
public:
    PipelineBase(PipelineType type, PipelineInfo info) : info(info), type(type) {}
    virtual ~PipelineBase() = default;

    // Disable copy constructor
    PipelineBase(const PipelineBase&) = delete;
    PipelineBase& operator=(const PipelineBase&) = delete;

    // Binds the texture in the specified slot
    virtual void BindTexture(u32 group, u32 slot, TextureHandle handle) = 0;

    // Binds the texture in the specified slot
    virtual void BindBuffer(u32 group, u32 slot, BufferHandle handle,
                            u32 offset = 0, u32 range = WHOLE_SIZE, u32 view = 0) = 0;

    // Binds the sampler in the specified slot
    virtual void BindSampler(u32 group, u32 slot, SamplerHandle handle) = 0;

    // Binds a small uniform block (under 256 bytes) to the current pipeline
    virtual void BindPushConstant(std::span<const std::byte> data) = 0;

    // Sets the viewport of the pipeline
    virtual void SetViewport(float x, float y, float width, float height) = 0;

    // Sets the scissor of the pipeline
    virtual void SetScissor(s32 x, s32 y, u32 width, u32 height) = 0;

    // Returns the pipeline type (Graphics or Compute)
    PipelineType GetType() const {
        return type;
    }

protected:
    const PipelineInfo info;
    PipelineType type = PipelineType::Graphics;
};

using PipelineHandle = IntrusivePtr<PipelineBase>;

} // namespace VideoCore
