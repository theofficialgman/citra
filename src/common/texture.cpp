// Copyright 2019 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <vector>
#include "common/assert.h"
#include "common/common_types.h"

namespace Common {
void FlipRGBA8Texture(std::vector<u8>& tex, u64 width, u64 height) {
    ASSERT(tex.size() == width * height * 4);
    const u64 line_size = width * 4;
    for (u64 line = 0; line < height / 2; line++) {
        const u64 offset_1 = line * line_size;
        const u64 offset_2 = (height - line - 1) * line_size;
        // Swap lines
        std::swap_ranges(tex.begin() + offset_1, tex.begin() + offset_1 + line_size,
                         tex.begin() + offset_2);
    }
}
} // namespace Common
