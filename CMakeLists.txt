cmake_minimum_required(VERSION 3.21)

project(ggraph)

# used from: https://github.com/abetlen/llama-cpp-python
function(llama_cpp_python_install_target target)
    if(NOT TARGET ${target})
        return()
    endif()

    install(
        TARGETS ${target}
        LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/ggraph/wrapper
        RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/ggraph/wrapper
        ARCHIVE DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/ggraph/wrapper
        FRAMEWORK DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/ggraph/wrapper
        RESOURCE DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/ggraph/wrapper
    )
    install(
        TARGETS ${target}
        LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/ggraph/wrapper
        RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}/ggraph/wrapper
        ARCHIVE DESTINATION ${SKBUILD_PLATLIB_DIR}/ggraph/wrapper
        FRAMEWORK DESTINATION ${SKBUILD_PLATLIB_DIR}/ggraph/wrapper
        RESOURCE DESTINATION ${SKBUILD_PLATLIB_DIR}/ggraph/wrapper
    )
    set_target_properties(${target} PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
    if(UNIX)
        if(APPLE)
            set_target_properties(${target} PROPERTIES
                INSTALL_RPATH "@loader_path"
                BUILD_WITH_INSTALL_RPATH TRUE
            )
        else()
            set_target_properties(${target} PROPERTIES
                INSTALL_RPATH "$ORIGIN"
                BUILD_WITH_INSTALL_RPATH TRUE
            )
        endif()
    endif()
endfunction()

set(BUILD_SHARED_LIBS "On")

set(CMAKE_SKIP_BUILD_RPATH FALSE)

# When building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

# Add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_SKIP_RPATH FALSE)

# Enable building of the common library
set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build llama.cpp common library" FORCE)

# Disable unneeded features
set(LLAMA_CURL OFF CACHE BOOL "llama.cpp: enable curl" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "llama.cpp: enable curl" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "llama.cpp: enable curl" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "llama.cpp: enable curl" FORCE)

# Build native optimized binaries
set(GGML_NATIVE ON CACHE BOOL "llama.cpp: enable curl" FORCE)

# TODO: auto detect if we should build the GPU + RPC backends

# Architecture detection and settings for Apple platforms
if (APPLE)
    # Get the target architecture
    execute_process(
        COMMAND uname -m
        OUTPUT_VARIABLE HOST_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # If CMAKE_OSX_ARCHITECTURES is not set, use the host architecture
    if(NOT CMAKE_OSX_ARCHITECTURES)
        set(CMAKE_OSX_ARCHITECTURES ${HOST_ARCH} CACHE STRING "Build architecture for macOS" FORCE)
    endif()

    message(STATUS "Host architecture: ${HOST_ARCH}")
    message(STATUS "Target architecture: ${CMAKE_OSX_ARCHITECTURES}")

    # Configure based on target architecture
    if(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
        # Intel Mac settings
        set(GGML_AVX "OFF" CACHE BOOL "ggml: enable AVX" FORCE)
        set(GGML_AVX2 "OFF" CACHE BOOL "ggml: enable AVX2" FORCE)
        set(GGML_FMA "OFF" CACHE BOOL "ggml: enable FMA" FORCE)
        set(GGML_F16C "OFF" CACHE BOOL "ggml: enable F16C" FORCE)
    endif()

    # Metal settings (enable for both architectures)
    set(GGML_METAL "ON" CACHE BOOL "ggml: enable Metal" FORCE)
    set(GGML_METAL_EMBED_LIBRARY "ON" CACHE BOOL "ggml: embed metal library" FORCE)
endif()


# Try to find CUDA + enable the backend if found
find_package(CUDAToolkit QUIET)

if (NOT CUDAToolkit_FOUND)
    message(STATUS "CUDA not found. Skipping CUDA backend.")
    set(GGML_CUDA OFF CACHE BOOL "ggml: enable CUDA" FORCE)
    
else()
    message(STATUS "Configuring for CUDA ${CUDAToolkit_VERSION_MAJOR}")
    set(GGML_CUDA ON CACHE BOOL "ggml: enable CUDA" FORCE)
    set(CMAKE_CUDA_ARCHITECTURES "all-major")
    set(CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets")

    if (CUDAToolkit_VERSION_MAJOR EQUAL 11 AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "10")
        message(FATAL_ERROR "CUDA 11.x requires GCC 10 or lower. Please re-install using 'CMAKE_ARGS=\"-DCMAKE_C_COMPILER=gcc-10 -DCMAKE_CXX_COMPILER=g++-10\"'")
    elseif (CUDAToolkit_VERSION_MAJOR EQUAL 12 AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "12")
        message(FATAL_ERROR "CUDA 12.x requires GCC 12 or lower. Please re-install using 'CMAKE_ARGS=\"-DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12\"'")
    endif()
endif()

# enable RPC backend
set(GGML_RPC "ON" CACHE BOOL "ggml: enable RPC" FORCE)

add_subdirectory(vendor/llama.cpp)
llama_cpp_python_install_target(llama)
llama_cpp_python_install_target(ggml)

llama_cpp_python_install_target(ggml-base)

llama_cpp_python_install_target(ggml-amx)
llama_cpp_python_install_target(ggml-blas)
llama_cpp_python_install_target(ggml-can)
llama_cpp_python_install_target(ggml-cpu)
llama_cpp_python_install_target(ggml-cuda)
llama_cpp_python_install_target(ggml-hip)
llama_cpp_python_install_target(ggml-kompute)
llama_cpp_python_install_target(ggml-metal)
llama_cpp_python_install_target(ggml-musa)
llama_cpp_python_install_target(ggml-rpc)
llama_cpp_python_install_target(ggml-sycl)
llama_cpp_python_install_target(ggml-vulkan)

# Workaround for Windows + CUDA https://github.com/abetlen/llama-cpp-python/issues/563
if (WIN32)
    install(
        FILES $<TARGET_RUNTIME_DLLS:llama>
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/ggraph/wrapper
    )
    install(
        FILES $<TARGET_RUNTIME_DLLS:llama>
        DESTINATION ${SKBUILD_PLATLIB_DIR}/ggraph/wrapper
    )
    install(
        FILES $<TARGET_RUNTIME_DLLS:ggml>
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/ggraph/wrapper
    )
    install(
        FILES $<TARGET_RUNTIME_DLLS:ggml>
        DESTINATION ${SKBUILD_PLATLIB_DIR}/ggraph/wrapper
    )
endif()