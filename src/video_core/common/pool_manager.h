// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/object_pool.h"
#include "common/intrusive_ptr.h"

namespace VideoCore {

// Manages (de)allocation of video backend resources
class PoolManager {
public:
    template <typename T, typename... P>
    IntrusivePtr<T> Allocate(P&&... p) {
        auto& pool = GetPoolForType<T>();
        return IntrusivePtr<T>{pool.Allocate(std::forward<P>(p)...)};
    }

    template <typename T>
    void Free(T* ptr) {
        auto& pool = GetPoolForType<T>();
        pool.Free(ptr);
    }

private:
    template <typename T>
    ObjectPool<T>& GetPoolForType() {
        static ObjectPool<T> resource_pool;
        return resource_pool;
    }
};

} // namespace VideoCore
