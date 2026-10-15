cmake_minimum_required(VERSION 3.20)
project(tensorcore_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (MSVC)
    add_compile_options(/W4 /EHsc)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

include_directories(include)

# Core library
file(GLOB_RECURSE ENGINE_SOURCES "src/*.cpp")
add_library(tensorcore STATIC ${ENGINE_SOURCES})

# Test executable
enable_testing()
add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE tensorcore)

add_test(NAME EngineUnitTests COMMAND test_tensor)
