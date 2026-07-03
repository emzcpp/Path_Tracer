cmake_minimum_required(VERSION 3.20)
project(pathtracer CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(pathtracer
    src/main.cpp
    src/image.cpp
    src/renderer.cpp
    src/gltf_loader.cpp
    src/bvh.cpp
    src/scene_io.cpp
)

target_include_directories(pathtracer PRIVATE external)

find_package(Threads REQUIRED)
target_link_libraries(pathtracer PRIVATE Threads::Threads)

if(APPLE)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 20)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

    # Embed the MSL kernel as a C array (compiled at runtime — the offline
    # Metal compiler needs full Xcode, which we don't require).
    set(PT_MSL_HEADER ${CMAKE_BINARY_DIR}/generated/pathtrace_msl.h)
    add_custom_command(
        OUTPUT ${PT_MSL_HEADER}
        COMMAND ${CMAKE_COMMAND}
                -DTYPES=${CMAKE_SOURCE_DIR}/src/kernel_types.h
                -DKERNEL=${CMAKE_SOURCE_DIR}/src/kernels/pathtrace.metal
                -DOUT=${PT_MSL_HEADER}
                -P ${CMAKE_SOURCE_DIR}/cmake/embed_msl.cmake
        DEPENDS ${CMAKE_SOURCE_DIR}/src/kernel_types.h
                ${CMAKE_SOURCE_DIR}/src/kernels/pathtrace.metal
                ${CMAKE_SOURCE_DIR}/cmake/embed_msl.cmake
        COMMENT "Embedding MSL kernel source")

    target_sources(pathtracer PRIVATE
        src/viewer_macos.mm
        src/gpu_renderer.mm
        ${PT_MSL_HEADER}
        external/imgui/imgui.cpp
        external/imgui/imgui_draw.cpp
        external/imgui/imgui_tables.cpp
        external/imgui/imgui_widgets.cpp
        external/imgui/backends/imgui_impl_osx.mm
        external/imgui/backends/imgui_impl_metal.mm
        external/imguizmo/ImGuizmo.cpp)
    set_source_files_properties(src/viewer_macos.mm src/gpu_renderer.mm
                                external/imgui/backends/imgui_impl_osx.mm
                                external/imgui/backends/imgui_impl_metal.mm
                                PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    target_include_directories(pathtracer PRIVATE ${CMAKE_BINARY_DIR}/generated
                                                  external/imgui
                                                  external/imgui/backends
                                                  external/imguizmo)
    # GameController: required by imgui_impl_osx (gamepad support).
    target_link_libraries(pathtracer PRIVATE "-framework Cocoa"
                                             "-framework QuartzCore"
                                             "-framework Metal"
                                             "-framework GameController")
    target_compile_definitions(pathtracer PRIVATE PT_HAVE_VIEWER=1
                                                  PT_HAVE_METAL=1)
endif()
