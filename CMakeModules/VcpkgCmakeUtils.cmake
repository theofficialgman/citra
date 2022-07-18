# Enable library versioning and manifests
set(VCPKG_FEATURE_FLAGS "manifests,versions" CACHE INTERNAL "Necessary vcpkg flags for manifest based autoinstall and versioning")

# enable rebuilding of packages if requested by changed configuration
if(NOT DEFINED VCPKG_RECURSE_REBUILD_FLAG)
    set(VCPKG_RECURSE_REBUILD_FLAG "--recurse" CACHE INTERNAL "Enable rebuilding of packages if requested by changed configuration by default")
endif()

# Configure vcpkg
set(VCPKG_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/externals/vcpkg")
if (WIN32)
    execute_process(COMMAND cmd /C "${VCPKG_DIRECTORY}/bootstrap-vcpkg.bat")
    set(VCPKG_EXECUTABLE "${VCPKG_DIRECTORY}/vcpkg.exe")
else()
    execute_process(COMMAND bash "${VCPKG_DIRECTORY}/bootstrap-vcpkg.sh")
    set(VCPKG_EXECUTABLE "${VCPKG_DIRECTORY}/vcpkg")
endif()

# Install a package using the command line interface
function(vcpkg_add_package PKG_NAME)
    # Run the executable to install dependencies
    set(VCPKG_TARGET_TRIPLET_FLAG "--triplet=${VCPKG_TARGET_TRIPLET}")
    message(STATUS "VCPKG: Installing ${PKG_NAME}")
    execute_process(COMMAND ${VCPKG_EXECUTABLE} ${VCPKG_TARGET_TRIPLET_FLAG} ${VCPKG_RECURSE_REBUILD_FLAG} --disable-metrics install "${PKG_NAME}" WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} RESULT_VARIABLE VCPKG_INSTALL_OK)
    if (NOT VCPKG_INSTALL_OK EQUAL "0")
        message(FATAL_ERROR "VCPKG: Failed installing ${PKG_NAME}!")
    endif()
endfunction()
