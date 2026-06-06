# CMake setup for CGP library project with OpenGL
cmake_minimum_required(VERSION 3.16)

project(project
    DESCRIPTION "Project"
    LANGUAGES CXX
)

# Collect sources automatically (src + cgp library + shaders)
file(GLOB_RECURSE SRC_FILES
    "${CMAKE_CURRENT_LIST_DIR}/src/*.[ch]pp"
    "${CMAKE_CURRENT_LIST_DIR}/cgp/*.[ch]pp"
    "${CMAKE_CURRENT_LIST_DIR}/shaders/*.glsl"
)

add_executable(${PROJECT_NAME} ${SRC_FILES})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    "${CMAKE_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/cgp"
)

# Compile definitions (macros)
target_compile_definitions(${PROJECT_NAME} PRIVATE
    SOLUTION
    CGP_OPENGL_4_3               # OpenGL version (alternatives: CGP_OPENGL_3_3, CGP_OPENGL_4_1, CGP_OPENGL_4_3, CGP_OPENGL_4_6)
    IMGUI_IMPL_OPENGL_LOADER_GLAD # ImGui uses GLAD for OpenGL loading
    CGP_ERROR_EXCEPTION           # Throw exceptions on errors (comment to abort instead)
    # CGP_NO_DEBUG                # Uncomment to disable assertions
)

# Compiler standard
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Option to use precompiled GLFW on macOS
option(MACOS_GLFW_PRECOMPILED "Use precompiled library for GLFW on MacOS" OFF)

# Compiler warnings
if(MSVC)
    set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
    set_target_properties(${PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}$<0:>
        VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${SRC_FILES})
    target_compile_options(${PROJECT_NAME} PRIVATE /W4 /permissive-
        /wd4127 /wd4267 /wd4244 /wd4996 /wd26451 /wd26495 /wd4305 /wd4458
    )
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
        -Wall -Wextra -pedantic
        -Wfatal-errors -Wno-pragmas -Wno-unknown-warning-option
        -Wno-sign-compare -Wno-type-limits
    )
endif()

# Link required libraries for Unix (Glad needs dl, OpenGL framework on macOS)
if(UNIX AND NOT EMSCRIPTEN)
    target_link_libraries(${PROJECT_NAME} PRIVATE dl)
endif()

# GLFW library setup (windowing for OpenGL)
if(UNIX AND NOT EMSCRIPTEN)
    if(APPLE AND MACOS_GLFW_PRECOMPILED)
        # Use precompiled GLFW (option for macOS)
        message(STATUS "Using precompiled GLFW for macOS")
        target_include_directories(${PROJECT_NAME} PRIVATE
            "${CMAKE_CURRENT_LIST_DIR}/cgp/third_party/glfw/macos/include"
        )
        target_link_libraries(${PROJECT_NAME} PRIVATE
            "${CMAKE_CURRENT_LIST_DIR}/cgp/third_party/glfw/macos/lib/libglfw.3.dylib"
        )
    else()
        # Use system GLFW (Linux/macOS via pkg-config)
        find_package(PkgConfig REQUIRED)
        pkg_search_module(GLFW REQUIRED glfw3)
        target_include_directories(${PROJECT_NAME} PRIVATE ${GLFW_INCLUDE_DIRS})
        target_link_directories(${PROJECT_NAME} PRIVATE ${GLFW_LIBRARY_DIRS})
        target_link_libraries(${PROJECT_NAME} PRIVATE ${GLFW_LIBRARIES})
    endif()
elseif(WIN32)
    # Use bundled GLFW for Windows
    target_include_directories(${PROJECT_NAME} PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/cgp/third_party/glfw/windows/include"
    )
    if(MSVC)
        target_link_libraries(${PROJECT_NAME} PRIVATE
            "${CMAKE_CURRENT_LIST_DIR}/cgp/third_party/glfw/windows/lib-visual/glfw3.lib"
        )
    else()
        target_link_libraries(${PROJECT_NAME} PRIVATE
            "${CMAKE_CURRENT_LIST_DIR}/cgp/third_party/glfw/windows/lib-mingw-w64/libglfw3.a"
        )
    endif()
endif()

# Emscripten/WebGL2 path is intentionally dropped: this project requires
# desktop OpenGL 4.3+ for compute shaders, SSBO atomics, and indirect dispatch.
if(EMSCRIPTEN)
    message(FATAL_ERROR "Emscripten/WebGL2 is no longer supported. Build with a desktop OpenGL 4.3+ toolchain.")
endif()

# Bullet Physics library
set(BUILD_BULLET2_DEMOS OFF CACHE BOOL "" FORCE)
set(BUILD_BULLET3 OFF CACHE BOOL "" FORCE)
set(BUILD_EXTRAS OFF CACHE BOOL "" FORCE)
set(BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_CPU_DEMOS OFF CACHE BOOL "" FORCE)
set(BUILD_ENET OFF CACHE BOOL "" FORCE)
set(BUILD_CLSOCKET OFF CACHE BOOL "" FORCE)
set(BUILD_OPENGL3_DEMOS OFF CACHE BOOL "" FORCE)
set(BUILD_PYBULLET OFF CACHE BOOL "" FORCE)
set(USE_GRAPHICAL_BENCHMARK OFF CACHE BOOL "" FORCE)
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/third_party/bullet3" EXCLUDE_FROM_ALL)

target_link_libraries(${PROJECT_NAME} PRIVATE
    BulletDynamics
    BulletCollision
    LinearMath
)
target_include_directories(${PROJECT_NAME} PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/third_party/bullet3/src"
)
