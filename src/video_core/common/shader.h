// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <span>
#include <string_view>
#include <vector>
#include "common/common_types.h"
#include "common/intrusive_ptr.h"

namespace VideoCore {

enum class ShaderStage : u32 {
    Vertex = 0,
    Geometry = 1,
    Fragment = 2,
    Compute = 3,
    Undefined = 4
};

// Tells the module how much to optimize the bytecode
enum class ShaderOptimization : u32 {
    High = 0,
    Debug = 1
};

struct ShaderDeleter;

// Compiles shader source to backend representation
class ShaderBase : public IntrusivePtrEnabled<ShaderBase, ShaderDeleter> {
public:
    ShaderBase(ShaderStage stage, std::string_view name, std::string&& source) :
        name(name), stage(stage), source(source) {}
    virtual ~ShaderBase() = default;

    // This method is called by ShaderDeleter. Forward to the derived pool!
    virtual void Free() = 0;

    // Compiles the shader source code
    virtual bool Compile(ShaderOptimization level) = 0;

    // Returns the name given the shader module
    std::string_view GetName() const {
        return name;
    }

    // Returns the pipeline stage the shader is assigned to
    ShaderStage GetStage() const {
        return stage;
    }

protected:
    std::string_view name = "None";
    ShaderStage stage = ShaderStage::Undefined;
    std::string source = "";
};

// Foward pointer to its parent pool
struct ShaderDeleter {
    void operator()(ShaderBase* shader) {
        shader->Free();
    }
};

using ShaderHandle = IntrusivePtr<ShaderBase>;

} // namespace VideoCore
