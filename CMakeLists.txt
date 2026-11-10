cmake_minimum_required(VERSION 3.18)

# arquitectura de la GPU (89 = NVIDIA RTX 4050 Ada Lovelace)
set(CMAKE_CUDA_ARCHITECTURES 89)

project(fem_simulator LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Optimizaciones matemáticas para C++ y CUDA
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -DNDEBUG -Xcompiler=-fPIC")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)

# Asegurar que el código sea independiente de posición para crear la librería compartida (.so)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ==================== CUDA Toolkit (cuSPARSE & cuBLAS) ====================
find_package(CUDAToolkit REQUIRED)

# ==================== pybind11 (submodule) ====================
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11/CMakeLists.txt")
    message(FATAL_ERROR "pybind11 was not found. Execute: git submodule update --init --recursive")
endif()

add_subdirectory(extern/pybind11)

# ==================== Project files ====================
set(SOURCES
    src/bindings.cpp
    src/matrixcsr.cpp
    src/physicsEngine.cpp
    src/solvers_cpu.cpp
    src/solvers_gpu.cu
    src/vector.cpp
)

# ==================== Build Python module ====================
pybind11_add_module(fem_simulator ${SOURCES})

target_include_directories(fem_simulator PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Enlazar las librerías matemáticas de alto rendimiento de NVIDIA
target_link_libraries(fem_simulator PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
)

# ==================== Output directories ====================
set_target_properties(fem_simulator PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin/$<CONFIG>"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin/$<CONFIG>"
)

# ==================== Installation ====================
install(TARGETS fem_simulator 
    DESTINATION ${CMAKE_SOURCE_DIR}/.venv/lib/python3.12/site-packages/
)

message(STATUS "fem_simulator (CUDA Accelerated) ready for compilation")
message(STATUS "src files: ${SOURCES}")
message(STATUS "Output → ${CMAKE_CURRENT_SOURCE_DIR}/bin/$<CONFIG>")