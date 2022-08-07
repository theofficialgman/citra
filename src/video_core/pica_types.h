// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <cmath>
#include <cstring>
#include <boost/serialization/access.hpp>
#include "common/common_types.h"

namespace Pica {

/**
 * Template class for converting arbitrary Pica float types to IEEE 754 32-bit single-precision
 * floating point.
 *
 * When decoding, format is as follows:
 *  - The first `M` bits are the mantissa
 *  - The next `E` bits are the exponent
 *  - The last bit is the sign bit
 *
 * @todo Verify on HW if this conversion is sufficiently accurate.
 */
template <u32 M, u32 E>
struct Float {
    static constexpr u32 width = M + E + 1;
    static constexpr u32 bias = 128 - (1 << (E - 1));
    static constexpr u32 exponent_mask = (1 << E) - 1;
    static constexpr u32 mantissa_mask = (1 << M) - 1;
    static constexpr u32 sign_mask = 1 << (E + M);
public:
    static Float FromFloat32(float val) {
        Float ret;
        ret.value = val;
        return ret;
    }

    static Float FromRaw(u32 hex) {
        Float res;

        u32 exponent = (hex >> M) & exponent_mask;
        const u32 mantissa = hex & mantissa_mask;
        const u32 sign = (hex & sign_mask) << (31 - M - E);

        if (hex & (mantissa_mask | (exponent_mask << M))) {
            if (exponent == exponent_mask) {
                exponent = 255;
            } else {
                exponent += bias;
            }

            hex = sign | (mantissa << (23 - M)) | (exponent << 23);
        } else {
            hex = sign;
        }

        std::memcpy(&res.value, &hex, sizeof(float));
        return res;
    }

    static Float Zero() {
        return FromFloat32(0.f);
    }

    // Not recommended for anything but logging
    float ToFloat32() const {
        return value;
    }

    Float operator*(const Float& flt) const {
        float result = value * flt.ToFloat32();
        // PICA gives 0 instead of NaN when multiplying by inf
        if (std::isnan(result) && !std::isnan(value) && !std::isnan(flt.ToFloat32())) {
            result = 0.f;
        }

        return Float::FromFloat32(result);
    }

    Float operator/(const Float& flt) const {
        return Float::FromFloat32(ToFloat32() / flt.ToFloat32());
    }

    Float operator+(const Float& flt) const {
        return Float::FromFloat32(ToFloat32() + flt.ToFloat32());
    }

    Float operator-(const Float& flt) const {
        return Float::FromFloat32(ToFloat32() - flt.ToFloat32());
    }

    Float& operator*=(const Float& flt) {
        value = operator*(flt).value;
        return *this;
    }

    Float& operator/=(const Float& flt) {
        value /= flt.ToFloat32();
        return *this;
    }

    Float& operator+=(const Float& flt) {
        value += flt.ToFloat32();
        return *this;
    }

    Float& operator-=(const Float& flt) {
        value -= flt.ToFloat32();
        return *this;
    }

    Float operator-() const {
        return Float::FromFloat32(-ToFloat32());
    }

    bool operator<(const Float& flt) const {
        return ToFloat32() < flt.ToFloat32();
    }

    bool operator>(const Float& flt) const {
        return ToFloat32() > flt.ToFloat32();
    }

    bool operator>=(const Float& flt) const {
        return ToFloat32() >= flt.ToFloat32();
    }

    bool operator<=(const Float& flt) const {
        return ToFloat32() <= flt.ToFloat32();
    }

    bool operator==(const Float& flt) const {
        return ToFloat32() == flt.ToFloat32();
    }

    bool operator!=(const Float& flt) const {
        return ToFloat32() != flt.ToFloat32();
    }

private:
    // Stored as a regular float, merely for convenience
    // TODO: Perform proper arithmetic on this!
    float value;

    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int file_version) {
        ar& value;
    }
};

using float24 = Float<16, 7>;
using float20 = Float<12, 7>;
using float16 = Float<10, 5>;

} // namespace Pica