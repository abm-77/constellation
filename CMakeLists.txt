cmake_minimum_required(VERSION 3.24)

project(constellation)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_FLAGS "-g")
set(CMAKE_CXX_FLAGS "-g")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bld)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
set(PROJECT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)

add_compile_definitions(ENABLE_ASSERT=1)
add_compile_definitions(DEBUG=1)

find_package(SDL2 REQUIRED)

include_directories(${SDL2_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

file(GLOB grimoire_src CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/*.h" "${PROJECT_SOURCE_DIR}/*.c")

add_executable(${PROJECT_NAME}  ${grimoire_src})
target_link_libraries(${PROJECT_NAME} ${SDL2_LIBRARIES})
