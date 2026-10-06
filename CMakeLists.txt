cmake_minimum_required(VERSION 3.14)
project(laya VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(LAYA_BUILD_CLI   "laya: build laya-cli"   ON)
set(LAYA_LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp" CACHE PATH "llama.cpp source tree")
set(LAYA_LLAMA_CPP_TAG "8df332d" CACHE STRING "llama.cpp commit to fetch when LAYA_LLAMA_CPP_DIR is missing")

# --- llama.cpp (CPU only, library targets only)
set(GGML_CUDA   OFF CACHE BOOL "" FORCE)
set(GGML_METAL  OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP      OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL        OFF CACHE BOOL "" FORCE)

if (EXISTS "${LAYA_LLAMA_CPP_DIR}/CMakeLists.txt")
    add_subdirectory(${LAYA_LLAMA_CPP_DIR} llama.cpp EXCLUDE_FROM_ALL)
    set(LAYA_LLAMA_SRC ${LAYA_LLAMA_CPP_DIR})
else()
    include(FetchContent)
    FetchContent_Declare(llama_cpp GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git GIT_TAG ${LAYA_LLAMA_CPP_TAG})
    FetchContent_GetProperties(llama_cpp)
    if (NOT llama_cpp_POPULATED)
        FetchContent_Populate(llama_cpp)
        add_subdirectory(${llama_cpp_SOURCE_DIR} ${llama_cpp_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
    set(LAYA_LLAMA_SRC ${llama_cpp_SOURCE_DIR})
endif()

# --- liblaya
add_library(laya src/laya.cpp src/laya-model.cpp src/tokenizer.cpp src/nfc.cpp)
target_include_directories(laya PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(laya PUBLIC llama ggml)
target_compile_features(laya PUBLIC cxx_std_17)
set_target_properties(laya PROPERTIES PUBLIC_HEADER include/laya.h)
if (BUILD_SHARED_LIBS)
    target_compile_definitions(laya PUBLIC LAYA_SHARED PRIVATE LAYA_BUILD)
    set_target_properties(laya PROPERTIES CXX_VISIBILITY_PRESET hidden)
endif()

# --- CLI
if (LAYA_BUILD_CLI)
    add_executable(laya-cli tools/laya-cli.cpp)
    target_include_directories(laya-cli PRIVATE ${LAYA_LLAMA_SRC}/vendor)
    target_link_libraries(laya-cli PRIVATE laya)
    install(TARGETS laya-cli RUNTIME DESTINATION bin)
endif()

install(TARGETS laya
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include)
