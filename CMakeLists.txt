# CMake 3.12 required for 20 to be a valid value for CXX_STANDARD
cmake_minimum_required(VERSION 3.12)
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.15)
    # Don't override the warning flags in MSVC:
    cmake_policy(SET CMP0092 NEW)
    # Allow selecting MSVC runtime library using CMAKE_MSVC_RUNTIME_LIBRARY.
    cmake_policy(SET CMP0091 NEW)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/externals/cmake-modules")
include(DownloadExternals)
include(CMakeDependentOption)

project(citra LANGUAGES C CXX ASM)

# Set bundled sdl2/qt as dependent options.
# OFF by default, but if ENABLE_SDL2 and MSVC are true then ON
option(ENABLE_SDL2 "Enable the SDL2 frontend" ON)

option(ENABLE_QT "Enable the Qt frontend" ON)
option(ENABLE_QT_TRANSLATION "Enable translations for the Qt frontend" OFF)
CMAKE_DEPENDENT_OPTION(CITRA_USE_BUNDLED_QT "Download bundled Qt binaries" ON "ENABLE_QT;MSVC" OFF)

option(ENABLE_WEB_SERVICE "Enable web services (telemetry, etc.)" ON)

option(ENABLE_CUBEB "Enables the cubeb audio backend" ON)

option(ENABLE_FFMPEG_AUDIO_DECODER "Enable FFmpeg audio (AAC) decoder" OFF)
option(ENABLE_FFMPEG_VIDEO_DUMPER "Enable FFmpeg video dumper" OFF)

if (ENABLE_FFMPEG_AUDIO_DECODER OR ENABLE_FFMPEG_VIDEO_DUMPER)
    set(ENABLE_FFMPEG ON)
endif()

option(USE_DISCORD_PRESENCE "Enables Discord Rich Presence" OFF)

CMAKE_DEPENDENT_OPTION(ENABLE_MF "Use Media Foundation decoder (preferred over FFmpeg)" ON "WIN32" OFF)

CMAKE_DEPENDENT_OPTION(COMPILE_WITH_DWARF "Add DWARF debugging information" ON "MINGW" OFF)

CMAKE_DEPENDENT_OPTION(ENABLE_FDK "Use FDK AAC decoder" OFF "NOT ENABLE_FFMPEG_AUDIO_DECODER;NOT ENABLE_MF" OFF)

if(NOT EXISTS ${PROJECT_SOURCE_DIR}/.git/hooks/pre-commit)
    message(STATUS "Copying pre-commit hook")
    file(COPY hooks/pre-commit
        DESTINATION ${PROJECT_SOURCE_DIR}/.git/hooks)
endif()

# Use ccache for android if available
# =======================================================================
if (NOT $ENV{NDK_CCACHE} EQUAL "")
    set(CCACHE_EXE $ENV{NDK_CCACHE})
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_EXE})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_EXE})
endif()

# Sanity check : Check that all submodules are present
# =======================================================================

function(check_submodules_present)
    file(READ "${PROJECT_SOURCE_DIR}/.gitmodules" gitmodules)
    string(REGEX MATCHALL "path *= *[^ \t\r\n]*" gitmodules ${gitmodules})
    foreach(module ${gitmodules})
        string(REGEX REPLACE "path *= *" "" module ${module})
        if (NOT EXISTS "${PROJECT_SOURCE_DIR}/${module}/.git")
            message(SEND_ERROR "Git submodule ${module} not found."
                    "Please run: git submodule update --init --recursive")
        endif()
    endforeach()
endfunction()
check_submodules_present()

configure_file(${PROJECT_SOURCE_DIR}/dist/compatibility_list/compatibility_list.qrc
               ${PROJECT_BINARY_DIR}/dist/compatibility_list/compatibility_list.qrc
               COPYONLY)
if (ENABLE_COMPATIBILITY_LIST_DOWNLOAD AND NOT EXISTS ${PROJECT_BINARY_DIR}/dist/compatibility_list/compatibility_list.json)
    message(STATUS "Downloading compatibility list for citra...")
    file(DOWNLOAD
        https://api.citra-emu.org/gamedb/
        "${PROJECT_BINARY_DIR}/dist/compatibility_list/compatibility_list.json" SHOW_PROGRESS)
endif()
if (EXISTS ${PROJECT_SOURCE_DIR}/dist/compatibility_list/compatibility_list.json)
    file(COPY "${PROJECT_SOURCE_DIR}/dist/compatibility_list/compatibility_list.json"
        DESTINATION "${PROJECT_BINARY_DIR}/dist/compatibility_list/")
endif()
if (NOT EXISTS ${PROJECT_BINARY_DIR}/dist/compatibility_list/compatibility_list.json)
    file(WRITE ${PROJECT_BINARY_DIR}/dist/compatibility_list/compatibility_list.json "")
endif()

# Detect current compilation architecture and create standard definitions
# =======================================================================

include(CheckSymbolExists)
function(detect_architecture symbol arch)
    if (NOT DEFINED ARCHITECTURE)
        set(CMAKE_REQUIRED_QUIET 1)
        check_symbol_exists("${symbol}" "" ARCHITECTURE_${arch})
        unset(CMAKE_REQUIRED_QUIET)

        # The output variable needs to be unique across invocations otherwise
        # CMake's crazy scope rules will keep it defined
        if (ARCHITECTURE_${arch})
            set(ARCHITECTURE "${arch}" PARENT_SCOPE)
            set(ARCHITECTURE_${arch} 1 PARENT_SCOPE)
            add_definitions(-DARCHITECTURE_${arch}=1)
        endif()
    endif()
endfunction()

if (NOT ENABLE_GENERIC)
    if (MSVC)
        detect_architecture("_M_AMD64" x86_64)
        detect_architecture("_M_IX86" x86)
        detect_architecture("_M_ARM" ARM)
        detect_architecture("_M_ARM64" ARM64)
    else()
        detect_architecture("__x86_64__" x86_64)
        detect_architecture("__i386__" x86)
        detect_architecture("__arm__" ARM)
        detect_architecture("__aarch64__" ARM64)
    endif()
endif()
if (NOT DEFINED ARCHITECTURE)
    set(ARCHITECTURE "GENERIC")
    set(ARCHITECTURE_GENERIC 1)
    add_definitions(-DARCHITECTURE_GENERIC=1)
endif()
message(STATUS "Target architecture: ${ARCHITECTURE}")


# Configure C++ standard
# ===========================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set up output paths for executable binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin/$<CONFIG>)

# System imported libraries
# ======================

# Enable vcpkg features
if (NOT DEFINED VCPKG_MANIFEST_FEATURES)
    if (ENABLE_SDL2)
        find_package(SDL2 CONFIG)

        if (NOT SDL2_FOUND)
            message(STATUS "SDL2 not found. Using vcpkg")
            list(APPEND VCPKG_MANIFEST_FEATURES "citra-sdl2")
        endif()
    endif()

    if (ENABLE_WEB_SERVICE)
        list(APPEND VCPKG_MANIFEST_FEATURES "citra-web-service")
    endif()

    if (USE_DISCORD_PRESENCE)
        list(APPEND VCPKG_MANIFEST_FEATURES "citra-discord")
    endif()

    if (ENABLE_FFMPEG)
        find_package(FFmpeg COMPONENTS avcodec avformat avutil swscale swresample)

        if (NOT FFMPEG_FOUND OR "${FFmpeg_avcodec_VERSION}" VERSION_LESS "57.48.101")
            message(STATUS "FFmpeg not found. Using vcpkg")
            list(APPEND VCPKG_MANIFEST_FEATURES "citra-ffmpeg")
        endif()
    endif()

    if (ENABLE_FDK)
        find_library(FDK_AAC fdk-aac DOC "The path to fdk_aac library")
        if(FDK_AAC STREQUAL "FDK_AAC-NOTFOUND")
            message(STATUS "FDK AAC not found. Using vcpkg")
            list(APPEND VCPKG_MANIFEST_FEATURES "citra-fdk-aac")
        endif()
    endif()

    # Cache the selected manifest features.
    set(VCPKG_MANIFEST_FEATURES ${VCPKG_MANIFEST_FEATURES} CACHE STRING "")
endif()

# Prefer static linkage on windows and MinGW
if (MINGW)
    set(VCPKG_TARGET_TRIPLET "x64-mingw-static")
elseif (WIN32)
    # We could also build a dynamic CRT and not require this flag but vcpkg
    # developers actively discourage this. See:
    # https://github.com/microsoft/vcpkg/issues/15122#issuecomment-745497720
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    set(VCPKG_TARGET_TRIPLET "x64-windows-static")
endif()

# Include vcpkg toolchain file
include(${CMAKE_SOURCE_DIR}/externals/vcpkg/scripts/buildsystems/vcpkg.cmake)

# Prefer the -pthread flag on Linux.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Boost
find_package(Boost REQUIRED COMPONENTS serialization)

# DiscordRPC
if (USE_DISCORD_PRESENCE)
    find_package(discord-rpc CONFIG REQUIRED)
endif()

if (ENABLE_WEB_SERVICE)
    # OpenSSL
    find_package(OpenSSL REQUIRED)

    # JSON
    find_package(nlohmann_json CONFIG REQUIRED)

    # httplib
    find_path(CPP_HTTPLIB_INCLUDE_DIRS "httplib.h")
    add_library(httplib INTERFACE)
    target_include_directories(httplib INTERFACE ${CPP_HTTPLIB_INCLUDE_DIRS})

    # cpp-jwt
    find_package(cpp-jwt CONFIG REQUIRED)
endif()

# (xperia64): Only use libyuv on Android b/c of build issues on Windows and mandatory JPEG
if(ANDROID)
    # libyuv
    find_package(libyuv CONFIG REQUIRED)
endif()

# Find the rest of our dependecies
set(REQUIRED_PACKAGES
    Catch2
    cryptopp
    fmt
    glad
    glm
    zstd
    unofficial-enet
    lodepng
    glslang
    xbyak
)

foreach(PACKAGE ${REQUIRED_PACKAGES})
    find_package(${PACKAGE} CONFIG REQUIRED)
endforeach()

if (ENABLE_SDL2)
    find_package(SDL2 CONFIG REQUIRED)
else()
    set(SDL2_FOUND NO)
endif()

if (ENABLE_QT)
    if (CITRA_USE_BUNDLED_QT)
        if ((MSVC_VERSION GREATER_EQUAL 1920 AND MSVC_VERSION LESS 1940) AND ARCHITECTURE_x86_64)
            set(QT_VER qt-5.12.12-msvc2017_64)
        else()
            message(FATAL_ERROR "No bundled Qt binaries for your toolchain. Disable CITRA_USE_BUNDLED_QT and provide your own.")
        endif()

        if (DEFINED QT_VER)
            download_bundled_external("qt/" ${QT_VER} QT_PREFIX)
        endif()

        set(QT_PREFIX_HINT HINTS "${QT_PREFIX}")
    else()
        # Passing an empty HINTS seems to cause default system paths to get ignored in CMake 2.8 so
        # make sure to not pass anything if we don't have one.
        set(QT_PREFIX_HINT)
    endif()

    find_package(Qt5 REQUIRED COMPONENTS Widgets Multimedia ${QT_PREFIX_HINT})

    if (ENABLE_QT_TRANSLATION)
        find_package(Qt5 REQUIRED COMPONENTS LinguistTools ${QT_PREFIX_HINT})
    endif()
endif()

# Ensure libusb is properly configured (based on dolphin libusb include)
if(NOT APPLE)
    include(FindPkgConfig)
    find_package(LibUSB)
endif()
if (NOT LIBUSB_FOUND)
    set(LIBUSB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/externals/libusb/libusb/libusb")
    set(LIBUSB_LIBRARIES usb)
endif()

if (ENABLE_FFMPEG)
    if(ENABLE_FFMPEG_VIDEO_DUMPER)
        set(FFmpeg_COMPONENTS "avcodec avformat avutil swscale swresample")
    else()
        set(FFmpeg_COMPONENTS "avcodec")
    endif()

    find_package(FFmpeg REQUIRED COMPONENTS ${FFmpeg_COMPONENTS})
endif()

if (ENABLE_FDK)
    find_package(fdk-aac CONFIG REQUIRED)
endif()

# Platform-specific library requirements
# ======================================

if (APPLE)
    # Umbrella framework for everything GUI-related
    find_library(COCOA_LIBRARY Cocoa)
    find_library(AVFOUNDATION_LIBRARY AVFoundation)
    set(PLATFORM_LIBRARIES ${COCOA_LIBRARY} ${AVFOUNDATION_LIBRARY} ${IOKIT_LIBRARY} ${COREVIDEO_LIBRARY})
elseif (WIN32)
    # WSAPoll and SHGetKnownFolderPath (AppData/Roaming) didn't exist before WinNT 6.x (Vista)
    add_definitions(-D_WIN32_WINNT=0x0600 -DWINVER=0x0600)
    set(PLATFORM_LIBRARIES winmm ws2_32)
    if (MINGW)
        # PSAPI is the Process Status API
        set(PLATFORM_LIBRARIES ${PLATFORM_LIBRARIES} psapi imm32 version)
    endif()
elseif (CMAKE_SYSTEM_NAME MATCHES "^(Linux|kFreeBSD|GNU|SunOS)$")
    set(PLATFORM_LIBRARIES rt)
endif()

# Setup a custom clang-format target (if clang-format can be found) that will run
# against all the src files. This should be used before making a pull request.
# =======================================================================

set(CLANG_FORMAT_POSTFIX "-10")
find_program(CLANG_FORMAT
    NAMES clang-format${CLANG_FORMAT_POSTFIX}
          clang-format
    PATHS ${PROJECT_BINARY_DIR}/externals)
# if find_program doesn't find it, try to download from externals
if (NOT CLANG_FORMAT)
    if (WIN32)
        message(STATUS "Clang format not found! Downloading...")
        set(CLANG_FORMAT "${PROJECT_BINARY_DIR}/externals/clang-format${CLANG_FORMAT_POSTFIX}.exe")
        file(DOWNLOAD
            https://github.com/citra-emu/ext-windows-bin/raw/master/clang-format${CLANG_FORMAT_POSTFIX}.exe
            "${CLANG_FORMAT}" SHOW_PROGRESS
            STATUS DOWNLOAD_SUCCESS)
        if (NOT DOWNLOAD_SUCCESS EQUAL 0)
            message(WARNING "Could not download clang format! Disabling the clang format target")
            file(REMOVE ${CLANG_FORMAT})
            unset(CLANG_FORMAT)
        endif()
    else()
        message(WARNING "Clang format not found! Disabling the clang format target")
    endif()
endif()

if (CLANG_FORMAT)
    set(SRCS ${PROJECT_SOURCE_DIR}/src)
    set(CCOMMENT "Running clang format against all the .h and .cpp files in src/")
    if (WIN32)
        add_custom_target(clang-format
            COMMAND powershell.exe -Command "Get-ChildItem '${SRCS}/*' -Include *.cpp,*.h,*.mm -Recurse | Foreach {&'${CLANG_FORMAT}' -i $_.fullname}"
            COMMENT ${CCOMMENT})
    elseif(MINGW)
        add_custom_target(clang-format
            COMMAND find `cygpath -u ${SRCS}` -iname *.h -o -iname *.cpp -o -iname *.mm | xargs `cygpath -u ${CLANG_FORMAT}` -i
            COMMENT ${CCOMMENT})
    else()
        add_custom_target(clang-format
            COMMAND find ${SRCS} -iname *.h -o -iname *.cpp -o -iname *.mm | xargs ${CLANG_FORMAT} -i
            COMMENT ${CCOMMENT})
    endif()
    unset(SRCS)
    unset(CCOMMENT)
endif()

# Include source code
# ===================

# This function should be passed a list of all files in a target. It will automatically generate
# file groups following the directory hierarchy, so that the layout of the files in IDEs matches the
# one in the filesystem.
function(create_target_directory_groups target_name)
    # Place any files that aren't in the source list in a separate group so that they don't get in
    # the way.
    source_group("Other Files" REGULAR_EXPRESSION ".")

    get_target_property(target_sources "${target_name}" SOURCES)

    foreach(file_name IN LISTS target_sources)
        get_filename_component(dir_name "${file_name}" PATH)
        # Group names use '\' as a separator even though the entire rest of CMake uses '/'...
        string(REPLACE "/" "\\" group_name "${dir_name}")
        source_group("${group_name}" FILES "${file_name}")
    endforeach()
endfunction()

# Gets a UTC timstamp and sets the provided variable to it
function(get_timestamp _var)
    string(TIMESTAMP timestamp UTC)
    set(${_var} "${timestamp}" PARENT_SCOPE)
endfunction()

# Prevent boost from linking against libs when building
add_definitions(-DBOOST_ERROR_CODE_HEADER_ONLY
    -DBOOST_SYSTEM_NO_LIB
    -DBOOST_DATE_TIME_NO_LIB
    -DBOOST_REGEX_NO_LIB
)

# generate git/build information
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REF_SPEC GIT_REV)
git_describe(GIT_DESC --always --long --dirty)
git_branch_name(GIT_BRANCH)
get_timestamp(BUILD_DATE)

enable_testing()
add_subdirectory(externals)

add_subdirectory(src)
add_subdirectory(dist/installer)


# Set citra-qt project or citra project as default StartUp Project in Visual Studio depending on whether QT is enabled or not
if(ENABLE_QT)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT citra-qt)
else()
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT citra)
endif()

# Installation instructions
# =========================

# Install freedesktop.org metadata files, following those specifications:
# http://standards.freedesktop.org/desktop-entry-spec/desktop-entry-spec-latest.html
# http://standards.freedesktop.org/icon-theme-spec/icon-theme-spec-latest.html
# http://standards.freedesktop.org/shared-mime-info-spec/shared-mime-info-spec-latest.html
if(ENABLE_QT AND UNIX AND NOT APPLE)
    install(FILES "${PROJECT_SOURCE_DIR}/dist/citra.desktop"
            DESTINATION "${CMAKE_INSTALL_PREFIX}/share/applications")
    install(FILES "${PROJECT_SOURCE_DIR}/dist/citra.svg"
            DESTINATION "${CMAKE_INSTALL_PREFIX}/share/icons/hicolor/scalable/apps")
    install(FILES "${PROJECT_SOURCE_DIR}/dist/citra.xml"
            DESTINATION "${CMAKE_INSTALL_PREFIX}/share/mime/packages")
endif()

if(UNIX)
    if(ENABLE_SDL2)
        install(FILES "${PROJECT_SOURCE_DIR}/dist/citra.6"
                DESTINATION "${CMAKE_INSTALL_PREFIX}/share/man/man6")
    endif()

    if (ENABLE_QT)
        install(FILES "${PROJECT_SOURCE_DIR}/dist/citra-qt.6"
                DESTINATION "${CMAKE_INSTALL_PREFIX}/share/man/man6")
    endif()
endif()
