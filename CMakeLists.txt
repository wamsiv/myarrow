cmake_minimum_required(VERSION 3.13.4)
project(my_arrrow LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")


list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")
find_package(Arrow REQUIRED)
include_directories(${Arrow_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
list(APPEND CUDA_LIBRARIES ${CUDA_CUDA_LIBRARY})

add_executable(arrow_test arrow_test.cpp)
# add_executable(arrow_buffer arrow_buffer.cpp)
add_executable(arrow_serialization arrow_serialization.cpp)
# add_executable(arrow_cuda_test arrow_cuda_test.cu)
add_executable(arrow_gpu_test arrow_gpu_test.cpp)
# add_executable(device_to_device_copy device_to_device_copy.cpp)
add_executable(device_to_device_cuda device_to_device_cuda.cu)

add_executable(dict_memo dict_memo.cpp)

target_link_libraries(arrow_test ${Arrow_LIBRARIES})
# target_link_libraries(arrow_buffer ${Arrow_LIBRARIES})
# target_link_libraries(arrow_cuda_test ${Arrow_LIBRARIES})
target_link_libraries(arrow_serialization ${Arrow_LIBRARIES})
target_link_libraries(arrow_gpu_test ${Arrow_LIBRARIES} ${Arrow_GPU_LIBRARIES})
# target_link_libraries(device_to_device_copy ${CUDA_LIBRARIES} ${Arrow_LIBRARIES} ${Arrow_GPU_LIBRARIES})
target_link_libraries(device_to_device_cuda ${CUDA_LIBRARIES} ${Arrow_LIBRARIES} ${Arrow_GPU_LIBRARIES})

target_link_libraries(dict_memo ${Arrow_LIBRARIES})
