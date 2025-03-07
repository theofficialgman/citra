// Copyright 2014 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <atomic>
#include <string>
#include <unordered_map>
#include <vector>
#include "common/common_types.h"
#include "core/hle/service/cam/cam_params.h"

namespace Settings {

enum class RendererBackend : u32 {
    OpenGL = 0,
    Vulkan = 1,
};

enum class InitClock {
    SystemTime = 0,
    FixedTime = 1,
};

enum class LayoutOption {
    Default,
    SingleScreen,
    LargeScreen,
    SideScreen,
    // Similiar to default, but better for mobile devices in portrait mode. Top screen in clamped to
    // the top of the frame, and the bottom screen is enlarged to match the top screen.
    MobilePortrait,
    // Similiar to LargeScreen, but better for mobile devices in landscape mode. The screens are
    // clamped to the top of the frame, and the bottom screen is a bit bigger.
    MobileLandscape,
};

enum class MicInputType {
    None,
    Real,
    Static,
};

enum class StereoRenderOption {
    Off,
    SideBySide,
    Anaglyph,
    Interlaced,
    ReverseInterlaced,
    CardboardVR
};

namespace NativeButton {
enum Values {
    A,
    B,
    X,
    Y,
    Up,
    Down,
    Left,
    Right,
    L,
    R,
    Start,
    Select,
    Debug,
    Gpio14,

    ZL,
    ZR,

    Home,

    NumButtons,
};

constexpr int BUTTON_HID_BEGIN = A;
constexpr int BUTTON_IR_BEGIN = ZL;
constexpr int BUTTON_NS_BEGIN = Home;

constexpr int BUTTON_HID_END = BUTTON_IR_BEGIN;
constexpr int BUTTON_IR_END = BUTTON_NS_BEGIN;
constexpr int BUTTON_NS_END = NumButtons;

constexpr int NUM_BUTTONS_HID = BUTTON_HID_END - BUTTON_HID_BEGIN;
constexpr int NUM_BUTTONS_IR = BUTTON_IR_END - BUTTON_IR_BEGIN;
constexpr int NUM_BUTTONS_NS = BUTTON_NS_END - BUTTON_NS_BEGIN;

static const std::array<const char*, NumButtons> mapping = {{
    "button_a",
    "button_b",
    "button_x",
    "button_y",
    "button_up",
    "button_down",
    "button_left",
    "button_right",
    "button_l",
    "button_r",
    "button_start",
    "button_select",
    "button_debug",
    "button_gpio14",
    "button_zl",
    "button_zr",
    "button_home",
}};
} // namespace NativeButton

namespace NativeAnalog {
enum Values {
    CirclePad,
    CStick,
    NumAnalogs,
};

static const std::array<const char*, NumAnalogs> mapping = {{
    "circle_pad",
    "c_stick",
}};
} // namespace NativeAnalog

struct InputProfile {
    std::string name;
    std::array<std::string, NativeButton::NumButtons> buttons;
    std::array<std::string, NativeAnalog::NumAnalogs> analogs;
    std::string motion_device;
    std::string touch_device;
    bool use_touch_from_button;
    int touch_from_button_map_index;
    std::string udp_input_address;
    u16 udp_input_port;
    u8 udp_pad_index;
};

struct TouchFromButtonMap {
    std::string name;
    std::vector<std::string> buttons;
};

struct Values {
    // CheckNew3DS
    bool is_new_3ds;

    // Controls
    InputProfile current_input_profile;       ///< The current input profile
    int current_input_profile_index;          ///< The current input profile index
    std::vector<InputProfile> input_profiles; ///< The list of input profiles
    std::vector<TouchFromButtonMap> touch_from_button_maps;

    // Core
    bool use_cpu_jit;
    int cpu_clock_percentage;

    // Data Storage
    bool use_virtual_sd;
    std::string nand_dir;
    std::string sdmc_dir;

    // System
    int region_value;
    InitClock init_clock;
    u64 init_time;

    // Renderer
    RendererBackend renderer_backend = RendererBackend::Vulkan;
    bool renderer_debug = true;
    bool use_gles;
    bool use_hw_renderer;
    bool use_hw_shader;
    bool separable_shader;
    bool use_disk_shader_cache;
    bool shaders_accurate_mul;
    bool use_shader_jit;
    u16 resolution_factor;
    bool use_frame_limit_alternate;
    u16 frame_limit;
    u16 frame_limit_alternate;
    std::string texture_filter_name;

    LayoutOption layout_option;
    bool swap_screen;
    bool upright_screen;
    bool custom_layout;
    u16 custom_top_left;
    u16 custom_top_top;
    u16 custom_top_right;
    u16 custom_top_bottom;
    u16 custom_bottom_left;
    u16 custom_bottom_top;
    u16 custom_bottom_right;
    u16 custom_bottom_bottom;

    float bg_red;
    float bg_green;
    float bg_blue;

    StereoRenderOption render_3d;
    std::atomic<u8> factor_3d;

    int cardboard_screen_size;
    int cardboard_x_shift;
    int cardboard_y_shift;

    bool filter_mode;
    std::string pp_shader_name;

    bool dump_textures;
    bool custom_textures;
    bool preload_textures;

    bool use_vsync_new;

    // Audio
    bool enable_dsp_lle;
    bool enable_dsp_lle_multithread;
    std::string sink_id;
    bool enable_audio_stretching;
    std::string audio_device_id;
    float volume;
    MicInputType mic_input_type;
    std::string mic_input_device;

    // Camera
    std::array<std::string, Service::CAM::NumCameras> camera_name;
    std::array<std::string, Service::CAM::NumCameras> camera_config;
    std::array<int, Service::CAM::NumCameras> camera_flip;

    // Debugging
    bool record_frame_times;
    bool use_gdbstub;
    u16 gdbstub_port;
    std::string log_filter;
    std::unordered_map<std::string, bool> lle_modules;

    // WebService
    bool enable_telemetry;
    std::string web_api_url;
    std::string citra_username;
    std::string citra_token;

    // Video Dumping
    std::string output_format;
    std::string format_options;

    std::string video_encoder;
    std::string video_encoder_options;
    u64 video_bitrate;

    std::string audio_encoder;
    std::string audio_encoder_options;
    u64 audio_bitrate;
} extern values;

// a special value for Values::region_value indicating that citra will automatically select a region
// value to fit the region lockout info of the game
static constexpr int REGION_VALUE_AUTO_SELECT = -1;

void Apply();
void LogSettings();

// Input profiles
void LoadProfile(int index);
void SaveProfile(int index);
void CreateProfile(std::string name);
void DeleteProfile(int index);
void RenameCurrentProfile(std::string new_name);
} // namespace Settings
