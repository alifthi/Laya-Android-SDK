cmake_minimum_required(VERSION 3.14)
project(laya-choice CXX)

set(LAYA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../.. CACHE PATH "laya-cpp root")

add_executable(choice main.cpp)
target_include_directories(choice PRIVATE ${LAYA_DIR}/include)
target_link_libraries(choice PRIVATE ${LAYA_DIR}/build/liblaya.so)
