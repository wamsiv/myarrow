#include <cstdio>
#include <iostream>
#include <exception>

#include <cuda.h>
#include <arrow/api.h>
#include <arrow/gpu/cuda_api.h>

// typedef unsigned long long DevicePtr;
// typedef int DeviceResult;

constexpr int32_t kGpuNumber = 0;

inline auto cuda_check_error = [&](CUresult status) {
    if (status != CUDA_SUCCESS)
    {
        throw std::runtime_error("Cuda Error.");
    }
};

inline auto check_arrow_status = [&](arrow::Status status) {
    if (!status.ok())
    {
        throw std::runtime_error("Error in arrow calls.");
    }
};

inline auto generate_schema = [&](auto *schema) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    fields.push_back(field("int_", arrow::int64()));
    *schema = arrow::schema(fields);
};

int main()
{
    const int size = 1 << 5;
    // std::cout << size << std::endl;
    int64_t *arr = new int64_t[size];
    for (int i = 0; i < size; ++i)
    {
        arr[i] = i;
        // std::cout << arr[i] << ", ";
    }
    // pointer to memory holding data in GPU
    CUdeviceptr device_ptr;
    // Initialize cuda driver
    cuda_check_error(cuInit(0));

    // Get handle to the device
    CUdevice device;
    cuda_check_error(cuDeviceGet(&device, 0));

    // create context
    CUcontext ctx;
    cuda_check_error(cuCtxCreate(&ctx, 0, device));

    // set context to current CPU thread
    cuda_check_error(cuCtxSetCurrent(ctx));

    // Allocate memory on GPU
    cuda_check_error(cuMemAlloc(&device_ptr, sizeof(int64_t) * size));

    // Copy data from host to device
    cuda_check_error(cuMemcpyHtoD(device_ptr, arr, sizeof(int64_t) * size));

    // delete host array buffer
    delete[] arr;

    // cuda_check_error(cuMemcpyDtoH(h_arr, device_ptr, sizeof(int64_t) * size));

    arrow::cuda::CudaDeviceManager *manager;
    arrow::cuda::CudaDeviceManager::GetInstance(&manager);
    std::shared_ptr<arrow::cuda::CudaContext> context;
    manager->GetContext(kGpuNumber, &context);

    std::shared_ptr<arrow::cuda::CudaBuffer> cuda_buffer;
    check_arrow_status(context->Allocate(sizeof(int64_t) * size, &cuda_buffer));
    // context->View((uint8_t *)device_ptr, sizeof(int64_t) * size, &cuda_buffer);

    check_arrow_status(cuda_buffer->CopyFromDevice(0, (void *)device_ptr, sizeof(int64_t) * size));

    // validate device array
    int64_t *h_arr = new int64_t[size];

    check_arrow_status(cuda_buffer->CopyToHost(0, sizeof(int64_t) * size, h_arr));

    for (int i = 0; i < size; ++i)
    {
        printf("%d, ", h_arr[i]);
    }
    printf("\n");

    // clean up
    delete[] h_arr;

    std::shared_ptr<arrow::RecordBatch> record_batch;
    std::shared_ptr<arrow::Schema> schema;
    generate_schema(&schema);
    std::shared_ptr<arrow::cuda::CudaIpcMemHandle> handle;
    check_arrow_status(cuda_buffer->ExportForIpc(&handle));

    arrow::cuda::ReadRecordBatch(schema, cuda_buffer, arrow::default_memory_pool(), &record_batch);
    std::cout << record_batch->num_rows() << std::endl;
    std::cout << record_batch->num_columns() << std::endl;

    // clean up
    cuda_check_error(cuMemFree(device_ptr));
}