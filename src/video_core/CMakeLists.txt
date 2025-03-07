add_library(video_core STATIC
    command_processor.cpp
    command_processor.h
    debug_utils/debug_utils.cpp
    debug_utils/debug_utils.h
    geometry_pipeline.cpp
    geometry_pipeline.h
    gpu_debugger.h
    pica.cpp
    pica.h
    pica_state.h
    primitive_assembly.cpp
    primitive_assembly.h
    rasterizer_interface.h
    pica_regs.inc
    pica.cpp
    pica.h
    regs.cpp
    regs.h
    regs_framebuffer.h
    regs_lighting.h
    regs_pipeline.h
    regs_rasterizer.h
    regs_shader.h
    regs_texturing.h
    renderer_base.cpp
    renderer_base.h
    common/backend.h
    common/buffer.h
    common/framebuffer.h
    common/pica_types.h
    common/pica_uniforms.cpp
    common/pica_uniforms.h
    common/pipeline_cache.cpp
    common/pipeline_cache.h
    common/pool_manager.h
    common/rasterizer.cpp
    common/rasterizer.h
    common/rasterizer_cache.cpp
    common/rasterizer_cache.h
    common/renderer.cpp
    common/renderer.h
    common/shader_runtime_cache.h
    common/shader_disk_cache.cpp
    common/shader_disk_cache.h
    common/shader_gen.cpp
    common/shader_gen.h
    common/shader.h
    common/surface_params.cpp
    common/surface_params.h
    common/texture.h
    common/pipeline.h
    #renderer_opengl/frame_dumper_opengl.cpp
    #renderer_opengl/frame_dumper_opengl.h
    #renderer_opengl/gl_rasterizer.cpp
    #renderer_opengl/gl_rasterizer.h
    #renderer_opengl/gl_rasterizer_cache.cpp
    #renderer_opengl/gl_rasterizer_cache.h
    #renderer_opengl/gl_resource_manager.cpp
    #renderer_opengl/gl_resource_manager.h
    #renderer_opengl/gl_shader_decompiler.cpp
    #renderer_opengl/gl_shader_decompiler.h
    #renderer_opengl/gl_shader_disk_cache.cpp
    #renderer_opengl/gl_shader_disk_cache.h
    #renderer_opengl/gl_shader_gen.cpp
    #renderer_opengl/gl_shader_gen.h
    #renderer_opengl/gl_shader_manager.cpp
    #renderer_opengl/gl_shader_manager.h
    #renderer_opengl/gl_shader_util.cpp
    #renderer_opengl/gl_shader_util.h
    #renderer_opengl/gl_state.cpp
    #renderer_opengl/gl_state.h
    #renderer_opengl/gl_stream_buffer.cpp
    #renderer_opengl/gl_stream_buffer.h
    #renderer_opengl/gl_surface_params.cpp
    #renderer_opengl/gl_surface_params.h
    #renderer_opengl/gl_vars.cpp
    #renderer_opengl/gl_vars.h
    #renderer_opengl/pica_to_gl.h
    #renderer_opengl/post_processing_opengl.cpp
    #renderer_opengl/post_processing_opengl.h
    #renderer_opengl/renderer_opengl.cpp
    #renderer_opengl/renderer_opengl.h
    #renderer_opengl/texture_downloader_es.cpp
    #renderer_opengl/texture_downloader_es.h
    #renderer_opengl/texture_filters/anime4k/anime4k_ultrafast.cpp
    #renderer_opengl/texture_filters/anime4k/anime4k_ultrafast.h
    #renderer_opengl/texture_filters/bicubic/bicubic.cpp
    #renderer_opengl/texture_filters/bicubic/bicubic.h
    #renderer_opengl/texture_filters/scale_force/scale_force.cpp
    #renderer_opengl/texture_filters/scale_force/scale_force.h
    #renderer_opengl/texture_filters/texture_filter_base.h
    #renderer_opengl/texture_filters/texture_filterer.cpp
    #renderer_opengl/texture_filters/texture_filterer.h
    #renderer_opengl/texture_filters/xbrz/xbrz_freescale.cpp
    #renderer_opengl/texture_filters/xbrz/xbrz_freescale.h
    #temporary, move these back in alphabetical order before merging
    #renderer_opengl/gl_format_reinterpreter.cpp
    #renderer_opengl/gl_format_reinterpreter.h
    renderer_vulkan/pica_to_vulkan.h
    renderer_vulkan/vk_backend.cpp
    renderer_vulkan/vk_backend.h
    renderer_vulkan/vk_buffer.cpp
    renderer_vulkan/vk_buffer.h
    renderer_vulkan/vk_common.cpp
    renderer_vulkan/vk_common.h
    #renderer_vulkan/vk_format_reinterpreter.cpp
    #renderer_vulkan/vk_format_reinterpreter.h
    #renderer_vulkan/vk_format_util.cpp
    #renderer_vulkan/vk_format_util.h
    renderer_vulkan/vk_framebuffer.cpp
    renderer_vulkan/vk_framebuffer.h
    renderer_vulkan/vk_instance.cpp
    renderer_vulkan/vk_instance.h
    renderer_vulkan/vk_pipeline.cpp
    renderer_vulkan/vk_pipeline.h
    renderer_vulkan/vk_platform.h
    renderer_vulkan/vk_renderpass_cache.cpp
    renderer_vulkan/vk_renderpass_cache.h
    renderer_vulkan/vk_shader_gen.cpp
    renderer_vulkan/vk_shader_gen.h
    renderer_vulkan/vk_shader.cpp
    renderer_vulkan/vk_shader.h
    renderer_vulkan/vk_swapchain.cpp
    renderer_vulkan/vk_swapchain.h
    renderer_vulkan/vk_task_scheduler.cpp
    renderer_vulkan/vk_task_scheduler.h
    renderer_vulkan/vk_texture.cpp
    renderer_vulkan/vk_texture.h
    shader/debug_data.h
    shader/shader.cpp
    shader/shader.h
    shader/shader_interpreter.cpp
    shader/shader_interpreter.h
    swrasterizer/clipper.cpp
    swrasterizer/clipper.h
    swrasterizer/framebuffer.cpp
    swrasterizer/framebuffer.h
    swrasterizer/lighting.cpp
    swrasterizer/lighting.h
    swrasterizer/proctex.cpp
    swrasterizer/proctex.h
    swrasterizer/rasterizer.cpp
    swrasterizer/rasterizer.h
    swrasterizer/swrasterizer.cpp
    swrasterizer/swrasterizer.h
    swrasterizer/texturing.cpp
    swrasterizer/texturing.h
    texture/etc1.cpp
    texture/etc1.h
    texture/texture_decode.cpp
    texture/texture_decode.h
    utils.h
    vertex_loader.cpp
    vertex_loader.h
    video_core.cpp
    video_core.h
)

set(SHADER_FILES
    renderer_opengl/depth_to_color.frag
    renderer_opengl/depth_to_color.vert
    renderer_opengl/ds_to_color.frag
    renderer_opengl/texture_filters/anime4k/refine.frag
    renderer_opengl/texture_filters/anime4k/x_gradient.frag
    renderer_opengl/texture_filters/anime4k/y_gradient.frag
    renderer_opengl/texture_filters/bicubic/bicubic.frag
    renderer_opengl/texture_filters/scale_force/scale_force.frag
    renderer_opengl/texture_filters/tex_coord.vert
    renderer_opengl/texture_filters/xbrz/xbrz_freescale.frag
    renderer_opengl/texture_filters/xbrz/xbrz_freescale.vert
)

include(${CMAKE_CURRENT_SOURCE_DIR}/generate_shaders.cmake)

foreach(shader_file ${SHADER_FILES})
    get_filename_component(shader_file_name ${shader_file} NAME)
    GetShaderHeaderFile(${shader_file_name})
    list(APPEND SHADER_HEADERS ${shader_header_file})
endforeach()

add_custom_target(shaders
    BYPRODUCTS ${SHADER_HEADERS}
    COMMAND "${CMAKE_COMMAND}" -P ${CMAKE_CURRENT_SOURCE_DIR}/generate_shaders.cmake
    SOURCES ${SHADER_FILES}
)
add_dependencies(video_core shaders)

target_include_directories(video_core PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

if(ARCHITECTURE_x86_64)
    target_sources(video_core
        PRIVATE
            shader/shader_jit_x64.cpp
            shader/shader_jit_x64_compiler.cpp

            shader/shader_jit_x64.h
            shader/shader_jit_x64_compiler.h
    )
endif()

create_target_directory_groups(video_core)

target_link_libraries(video_core PUBLIC common core)

target_link_libraries(video_core PRIVATE nihstro-headers glad::glad glm::glm Boost::serialization)

# Include Vulkan headers
target_include_directories(video_core PRIVATE ../../externals/Vulkan-Headers/include)
target_include_directories(video_core PRIVATE ../../externals/vma/include)
target_link_libraries(video_core PRIVATE glslang SPIRV)

if (ARCHITECTURE_x86_64)
    target_link_libraries(video_core PUBLIC xbyak::xbyak)
endif()
