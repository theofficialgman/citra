// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <tuple>
#include <boost/icl/interval_map.hpp>
#include <boost/icl/interval_set.hpp>
#include <unordered_map>
#include <boost/functional/hash.hpp>
#include "common/assert.h"
#include "common/common_funcs.h"
#include "common/common_types.h"
#include "core/custom_tex_cache.h"
#include "video_core/renderer_vulkan/vk_surface_params.h"
#include "video_core/renderer_vulkan/vk_texture.h"
#include "video_core/texture/texture_decode.h"

namespace Vulkan {

class RasterizerCacheVulkan;
class TextureFilterer;
class FormatReinterpreterVulkan;

vk::Format GetFormatTuple(SurfaceParams::PixelFormat pixel_format);

struct HostTextureTag {
    vk::Format format;
    u32 width;
    u32 height;
    bool operator==(const HostTextureTag& rhs) const noexcept {
        return std::tie(format, width, height) == std::tie(rhs.format, rhs.width, rhs.height);
    };
};

struct TextureCubeConfig {
    PAddr px;
    PAddr nx;
    PAddr py;
    PAddr ny;
    PAddr pz;
    PAddr nz;
    u32 width;
    Pica::TexturingRegs::TextureFormat format;

    bool operator==(const TextureCubeConfig& rhs) const {
        return std::tie(px, nx, py, ny, pz, nz, width, format) ==
               std::tie(rhs.px, rhs.nx, rhs.py, rhs.ny, rhs.pz, rhs.nz, rhs.width, rhs.format);
    }

    bool operator!=(const TextureCubeConfig& rhs) const {
        return !(*this == rhs);
    }
};

} // namespace Vulkan

namespace std {
template <>
struct hash<Vulkan::HostTextureTag> {
    std::size_t operator()(const Vulkan::HostTextureTag& tag) const noexcept {
        std::size_t hash = 0;
        boost::hash_combine(hash, tag.format);
        boost::hash_combine(hash, tag.width);
        boost::hash_combine(hash, tag.height);
        return hash;
    }
};

template <>
struct hash<Vulkan::TextureCubeConfig> {
    std::size_t operator()(const Vulkan::TextureCubeConfig& config) const noexcept {
        std::size_t hash = 0;
        boost::hash_combine(hash, config.px);
        boost::hash_combine(hash, config.nx);
        boost::hash_combine(hash, config.py);
        boost::hash_combine(hash, config.ny);
        boost::hash_combine(hash, config.pz);
        boost::hash_combine(hash, config.nz);
        boost::hash_combine(hash, config.width);
        boost::hash_combine(hash, static_cast<u32>(config.format));
        return hash;
    }
};
} // namespace std

namespace Vulkan {

using SurfaceSet = std::set<Surface>;

using SurfaceRegions = boost::icl::interval_set<PAddr, std::less, SurfaceInterval>;
using SurfaceMap =
    boost::icl::interval_map<PAddr, Surface, boost::icl::partial_absorber, std::less,
                             boost::icl::inplace_plus, boost::icl::inter_section, SurfaceInterval>;
using SurfaceCache =
    boost::icl::interval_map<PAddr, SurfaceSet, boost::icl::partial_absorber, std::less,
                             boost::icl::inplace_plus, boost::icl::inter_section, SurfaceInterval>;

static_assert(std::is_same<SurfaceRegions::interval_type, SurfaceCache::interval_type>() &&
                  std::is_same<SurfaceMap::interval_type, SurfaceCache::interval_type>(),
              "incorrect interval types");

using SurfaceRect_Tuple = std::tuple<Surface, Common::Rectangle<u32>>;
using SurfaceSurfaceRect_Tuple = std::tuple<Surface, Surface, Common::Rectangle<u32>>;

using PageMap = boost::icl::interval_map<u32, int>;

enum class ScaleMatch {
    Exact,   // only accept same res scale
    Upscale, // only allow higher scale than params
    Ignore   // accept every scaled res
};

/**
 * A watcher that notifies whether a cached surface has been changed. This is useful for caching
 * surface collection objects, including texture cube and mipmap.
 */
struct SurfaceWatcher {
public:
    explicit SurfaceWatcher(std::weak_ptr<CachedSurface>&& surface) : surface(std::move(surface)) {}

    /**
     * Checks whether the surface has been changed.
     * @return false if the surface content has been changed since last Validate() call or has been
     * destroyed; otherwise true
     */
    bool IsValid() const {
        return !surface.expired() && valid;
    }

    /// Marks that the content of the referencing surface has been updated to the watcher user.
    void Validate() {
        ASSERT(!surface.expired());
        valid = true;
    }

    /// Gets the referencing surface. Returns null if the surface has been destroyed
    Surface Get() const {
        return surface.lock();
    }

private:
    friend struct CachedSurface;
    std::weak_ptr<CachedSurface> surface;
    bool valid = false;
};

class RasterizerCacheVulkan;

struct CachedSurface : SurfaceParams, std::enable_shared_from_this<CachedSurface> {
    CachedSurface(RasterizerCacheVulkan& owner) : owner{owner} {}
    ~CachedSurface();

    bool CanFill(const SurfaceParams& dest_surface, SurfaceInterval fill_interval) const;
    bool CanCopy(const SurfaceParams& dest_surface, SurfaceInterval copy_interval) const;

    bool IsRegionValid(SurfaceInterval interval) const {
        return (invalid_regions.find(interval) == invalid_regions.end());
    }

    bool IsSurfaceFullyInvalid() const {
        auto interval = GetInterval();
        return *invalid_regions.equal_range(interval).first == interval;
    }

    bool registered = false;
    SurfaceRegions invalid_regions;

    u32 fill_size = 0; /// Number of bytes to read from fill_data
    std::array<u8, 4> fill_data;

    Texture texture;

    /// max mipmap level that has been attached to the texture
    u32 max_level = 0;
    /// level_watchers[i] watches the (i+1)-th level mipmap source surface
    std::array<std::shared_ptr<SurfaceWatcher>, 7> level_watchers;

    bool is_custom = false;
    Core::CustomTexInfo custom_tex_info;

    static constexpr unsigned int GetBytesPerPixel(PixelFormat format) {
        return format == PixelFormat::Invalid
                   ? 0
                   : (format == PixelFormat::D24 || GetFormatType(format) == SurfaceType::Texture)
                         ? 4
                         : SurfaceParams::GetFormatBpp(format) / 8;
    }

    std::vector<u8> vk_buffer;

    // Read/Write data in 3DS memory to/from gl_buffer
    void LoadGPUBuffer(PAddr load_start, PAddr load_end);
    void FlushGPUBuffer(PAddr flush_start, PAddr flush_end);

    // Upload/Download data in vk_buffer in/to this surface's texture
    void UploadGPUTexture(Common::Rectangle<u32> rect);
    void DownloadGPUTexture(const Common::Rectangle<u32>& rect);

    std::shared_ptr<SurfaceWatcher> CreateWatcher() {
        auto watcher = std::make_shared<SurfaceWatcher>(weak_from_this());
        watchers.push_front(watcher);
        return watcher;
    }

    void InvalidateAllWatcher() {
        for (const auto& watcher : watchers) {
            if (auto locked = watcher.lock()) {
                locked->valid = false;
            }
        }
    }

    void UnlinkAllWatcher() {
        for (const auto& watcher : watchers) {
            if (auto locked = watcher.lock()) {
                locked->valid = false;
                locked->surface.reset();
            }
        }
        watchers.clear();
    }

private:
    RasterizerCacheVulkan& owner;
    std::list<std::weak_ptr<SurfaceWatcher>> watchers;
};

struct CachedTextureCube {
    Texture texture;
    u16 res_scale = 1;
    std::shared_ptr<SurfaceWatcher> px;
    std::shared_ptr<SurfaceWatcher> nx;
    std::shared_ptr<SurfaceWatcher> py;
    std::shared_ptr<SurfaceWatcher> ny;
    std::shared_ptr<SurfaceWatcher> pz;
    std::shared_ptr<SurfaceWatcher> nz;
};

class TextureDownloader;

class RasterizerCacheVulkan : NonCopyable {
public:
    RasterizerCacheVulkan();
    ~RasterizerCacheVulkan();

    /// Blit one surface's texture to another
    bool BlitSurfaces(const Surface& src_surface, const Common::Rectangle<u32>& src_rect,
                      const Surface& dst_surface, const Common::Rectangle<u32>& dst_rect);

    /// Copy one surface's region to another
    void CopySurface(const Surface& src_surface, const Surface& dst_surface,
                     SurfaceInterval copy_interval);

    /// Load a texture from 3DS memory to OpenGL and cache it (if not already cached)
    Surface GetSurface(const SurfaceParams& params, ScaleMatch match_res_scale,
                       bool load_if_create);

    /// Attempt to find a subrect (resolution scaled) of a surface, otherwise loads a texture from
    /// 3DS memory to OpenGL and caches it (if not already cached)
    SurfaceRect_Tuple GetSurfaceSubRect(const SurfaceParams& params, ScaleMatch match_res_scale,
                                        bool load_if_create);

    /// Get a surface based on the texture configuration
    Surface GetTextureSurface(const Pica::TexturingRegs::FullTextureConfig& config);
    Surface GetTextureSurface(const Pica::Texture::TextureInfo& info, u32 max_level = 0);

    /// Get the color and depth surfaces based on the framebuffer configuration
    SurfaceSurfaceRect_Tuple GetFramebufferSurfaces(bool using_color_fb, bool using_depth_fb,
                                                    const Common::Rectangle<s32>& viewport_rect);

    /// Get a surface that matches the fill config
    Surface GetFillSurface(const GPU::Regs::MemoryFillConfig& config);

    /// Get a surface that matches a "texture copy" display transfer config
    SurfaceRect_Tuple GetTexCopySurface(const SurfaceParams& params);

    /// Write any cached resources overlapping the region back to memory (if dirty)
    void FlushRegion(PAddr addr, u32 size, Surface flush_surface = nullptr);

    /// Mark region as being invalidated by region_owner (nullptr if 3DS memory)
    void InvalidateRegion(PAddr addr, u32 size, const Surface& region_owner);

    /// Flush all cached resources tracked by this cache manager
    void FlushAll();

    /// Clear all cached resources tracked by this cache manager
    void ClearAll(bool flush);

    // Textures from destroyed surfaces are stored here to be recyled to reduce allocation overhead
    // in the driver
    // this must be placed above the surface_cache to ensure all cached surfaces are destroyed
    // before destroying the recycler
    std::unordered_multimap<HostTextureTag, Texture> host_texture_recycler;

private:
    void DuplicateSurface(const Surface& src_surface, const Surface& dest_surface);

    /// Update surface's texture for given region when necessary
    void ValidateSurface(const Surface& surface, PAddr addr, u32 size);

    // Returns false if there is a surface in the cache at the interval with the same bit-width,
    bool NoUnimplementedReinterpretations(const Vulkan::Surface& surface,
                                          Vulkan::SurfaceParams& params,
                                          const Vulkan::SurfaceInterval& interval);

    // Return true if a surface with an invalid pixel format exists at the interval
    bool IntervalHasInvalidPixelFormat(SurfaceParams& params, const SurfaceInterval& interval);

    // Attempt to find a reinterpretable surface in the cache and use it to copy for validation
    bool ValidateByReinterpretation(const Surface& surface, SurfaceParams& params,
                                    const SurfaceInterval& interval);

    /// Create a new surface
    Surface CreateSurface(const SurfaceParams& params);

    /// Register surface into the cache
    void RegisterSurface(const Surface& surface);

    /// Remove surface from the cache
    void UnregisterSurface(const Surface& surface);

    /// Increase/decrease the number of surface in pages touching the specified region
    void UpdatePagesCachedCount(PAddr addr, u32 size, int delta);

    SurfaceCache surface_cache;
    PageMap cached_pages;
    SurfaceMap dirty_regions;
    SurfaceSet remove_surfaces;

    u16 resolution_scale_factor;

    std::unordered_map<TextureCubeConfig, CachedTextureCube> texture_cube_cache;

    std::recursive_mutex mutex;

public:
    void AllocateTexture(Texture& target, SurfaceParams::SurfaceType type, vk::Format format,
                                     u32 width, u32 height);
    std::unique_ptr<FormatReinterpreterVulkan> format_reinterpreter;
};

} // namespace OpenGL
