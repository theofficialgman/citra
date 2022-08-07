/* Copyright (c) 2017-2022 Hans-Kristian Arntzen
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <cstddef>
#include <utility>
#include <memory>
#include <atomic>
#include <type_traits>

/// Simple reference counter for single threaded environments
class SingleThreadCounter {
public:
    inline void AddRef() {
        count++;
    }

    inline bool Release() {
        return --count == 0;
    }

private:
    std::size_t count = 1;
};

/// Thread-safe reference counter with atomics
class MultiThreadCounter {
public:
    MultiThreadCounter() {
        count.store(1, std::memory_order_relaxed);
    }

    inline void AddRef() {
        count.fetch_add(1, std::memory_order_relaxed);
    }

    inline bool Release() {
        auto result = count.fetch_sub(1, std::memory_order_acq_rel);
        return result == 1;
    }

private:
    std::atomic_size_t count;
};

template <typename T>
class IntrusivePtr;

template <typename T, typename Deleter = std::default_delete<T>,
          typename ReferenceOps = SingleThreadCounter>
class IntrusivePtrEnabled {
public:
    using IntrusivePtrType = IntrusivePtr<T>;
    using EnabledBase = T;
    using EnabledDeleter = Deleter;
    using EnabledReferenceOp = ReferenceOps;

    IntrusivePtrEnabled() = default;
    IntrusivePtrEnabled(const IntrusivePtrEnabled &) = delete;
    void operator=(const IntrusivePtrEnabled &) = delete;

    /// Decrement the reference counter and optionally free the memory
    inline void ReleaseRef() {
        if (ref_counter.Release()) {
            Deleter()(static_cast<T*>(this));
        }
    }

    /// Increment the reference counter
    inline void AddRef() {
        ref_counter.AddRef();
    }

protected:
    IntrusivePtr<T> RefFromThis();

private:
    ReferenceOps ref_counter;
};

/**
 * Lightweight alternative to std::shared_ptr for reference counting
 * usecases
 */
template <typename T>
class IntrusivePtr {
    using ReferenceBase = IntrusivePtrEnabled<
            typename T::EnabledBase,
            typename T::EnabledDeleter,
            typename T::EnabledReferenceOp>;

    template <typename U>
    friend class IntrusivePtr;
public:
    IntrusivePtr() = default;
    explicit IntrusivePtr(T *handle) : data(handle) {}

    template <typename U>
    IntrusivePtr(const IntrusivePtr<U> &other) {
        *this = other;
    }

    IntrusivePtr(const IntrusivePtr &other) {
        *this = other;
    }

    template <typename U>
    IntrusivePtr(IntrusivePtr<U> &&other) noexcept {
        *this = std::move(other);
    }

    IntrusivePtr(IntrusivePtr &&other) noexcept {
        *this = std::move(other);
    }

    ~IntrusivePtr() {
        Reset();
    }

    /// Returns a reference to the underlying data
    T& operator*() {
        return *data;
    }

    /// Returns an immutable reference to the underlying data
    const T& operator*() const {
        return *data;
    }

    /// Returns a pointer to the underlying data
    T* operator->() {
        return data;
    }

    /// Returns an immutable pointer to the underlying data
    const T* operator->() const {
        return data;
    }

    /// Returns true if the underlaying pointer it valid
    bool IsValid() const {
        return data != nullptr;
    }

    /// Default comparison operators
    auto operator<=>(const IntrusivePtr& other) const = default;

    /// Returns the raw pointer to the data
    T* Get() {
        return data;
    }

    /// Returns an immutable raw pointer to the data
    const T* Get() const {
        return data;
    }

    void Reset() {
        // Static up-cast here to avoid potential issues with multiple intrusive inheritance.
        // Also makes sure that the pointer type actually inherits from this type.
        if (data)
            static_cast<ReferenceBase*>(data)->ReleaseRef();
        data = nullptr;
    }

    template <typename U>
    IntrusivePtr& operator=(const IntrusivePtr<U>& other) {
        static_assert(std::is_base_of_v<T, U>, "Cannot safely assign downcasted intrusive pointers.");

        Reset();
        data = static_cast<T*>(other.data);

        // Static up-cast here to avoid potential issues with multiple intrusive inheritance.
        // Also makes sure that the pointer type actually inherits from this type.
        if (data) {
            static_cast<ReferenceBase*>(data)->ReleaseRef();
        }

        return *this;
    }

    IntrusivePtr& operator=(const IntrusivePtr& other) {
        if (this != &other) {
            Reset();
            data = other.data;
            if (data)
                static_cast<ReferenceBase*>(data)->AddRef();
        }

        return *this;
    }

    template <typename U>
    IntrusivePtr &operator=(IntrusivePtr<U> &&other) noexcept {
        Reset();
        data = std::exchange(other.data, nullptr);
        return *this;
    }

    IntrusivePtr &operator=(IntrusivePtr &&other) noexcept {
        if (this != &other) {
            Reset();
            data = std::exchange(other.data, nullptr);
        }

        return *this;
    }

    T* Release() & {
        return std::exchange(data, nullptr);
    }

    T* Release() && {
        return std::exchange(data, nullptr);
    }

private:
    T* data = nullptr;
};

template <typename T, typename Deleter, typename ReferenceOps>
IntrusivePtr<T> IntrusivePtrEnabled<T, Deleter, ReferenceOps>::RefFromThis() {
    AddRef();
    return IntrusivePtr<T>(static_cast<T*>(this));
}

template <typename Derived>
using DerivedIntrusivePtrType = IntrusivePtr<Derived>;

template <typename T, typename... P>
DerivedIntrusivePtrType<T> MakeHandle(P &&... p) {
    return DerivedIntrusivePtrType<T>(new T(std::forward<P>(p)...));
}

template <typename Base, typename Derived, typename... P>
typename Base::IntrusivePtrType MakeDerivedHandle(P &&... p) {
    return typename Base::IntrusivePtrType(new Derived(std::forward<P>(p)...));
}

template <typename T>
using ThreadSafeIntrusivePtrEnabled = IntrusivePtrEnabled<T, std::default_delete<T>, MultiThreadCounter>;
