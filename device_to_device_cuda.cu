#include <cstdio>
#include <exception>
#include <iostream>

// #include <arrow/api.h>
// #include <cuda.h>
#include <arrow/gpu/cuda_api.h>
// use runtime functions

// typedef unsigned long long DevicePtr;
// typedef int DeviceResult;

__global__ void addOneToArray(int64_t *device_ptr, const int size) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;

  // #ifdef __CUDA_ARCH__
  //   printf("Device Thread %d\n", threadIdx.x);
  // #else
  //   printf("Host code!\n");
  // #endif

  for (int64_t i = idx; i < size; i += stride) {
    device_ptr[i] += 1;
  }
}

inline __host__ void cuda_check_error(cudaError status) {
  if (status != cudaSuccess) {
    throw std::runtime_error("Cuda Error.");
  }
}

int main() {
  const int size = 1 << 5;
  // std::cout << size << std::endl;
  int64_t *arr = new int64_t[size];
  for (int i = 0; i < size; ++i) {
    arr[i] = i;
  }
  // pointer to memory holding data in GPU
  int64_t *device_ptr;

  // Allocate memory on GPU
  cuda_check_error(cudaMalloc((void **)&device_ptr, sizeof(int64_t) * size));

  // Copy data from host to device
  cuda_check_error(cudaMemcpy(device_ptr, arr, sizeof(int64_t) * size,
                              cudaMemcpyHostToDevice));

  // delete host array buffer
  delete[] arr;

  // manipulate array
  int64_t blocksize = 256;
  int64_t numblocks = (size + blocksize - 1) / blocksize;

  addOneToArray<<<numblocks, blocksize>>>(device_ptr, size);
  cuda_check_error(cudaDeviceSynchronize());

  // validate device array
  int64_t *h_arr = new int64_t[size];

  cuda_check_error(cudaMemcpy(h_arr, device_ptr, sizeof(int64_t) * size,
                              cudaMemcpyDeviceToHost));

  for (int i = 0; i < size; ++i) {
    printf("%d, ", h_arr[i]);
  }
  printf("\n");

  delete[] h_arr;
  cuda_check_error(cudaFree(device_ptr));
}