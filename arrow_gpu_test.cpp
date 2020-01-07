// #include "arrow/gpu/cuda_context.h"
#include "arrow/api.h"
#include "arrow/gpu/cuda_api.h"

#include <cstdint>
#include <cstdio>
#include <iostream>

using namespace arrow;

constexpr int32_t kGpuNumber = 0;

void getCpuData(std::shared_ptr<RecordBatch> &record_batch)
{
  std::vector<int64_t> arr = {1, 2, 3, 4, 5};
  std::vector<bool> is_null = {1, 1, 1, 1, 1};

  std::shared_ptr<Array> arrow_array;
  arrow::Int64Builder builder;

  builder.Resize(5);
  builder.AppendValues(arr, is_null);

  if (!builder.Finish(&arrow_array).ok())
  {
    throw std::runtime_error("Unsuccessful array allocation.");
  }

  std::vector<std::shared_ptr<Field>> fields;
  fields.push_back(field("int_", int64()));

  std::shared_ptr<Schema> schema = arrow::schema(fields);

  record_batch = arrow::RecordBatch::Make(schema, 5, {arrow_array});
}

int main()
{
  cuda::CudaDeviceManager *manager;
  std::shared_ptr<cuda::CudaContext> context;

  cuda::CudaDeviceManager::GetInstance(&manager);
  manager->GetContext(kGpuNumber, &context);

  std::shared_ptr<RecordBatch> record_batch;
  getCpuData(record_batch);

  //   std::cout << record_batch->num_rows() << std::endl;

  std::shared_ptr<cuda::CudaBuffer> device_serialized;

  cuda::SerializeRecordBatch(*record_batch, context.get(), &device_serialized);

  //   std::cout << "stop." << std::endl;

  std::shared_ptr<RecordBatch> read_batch;
  cuda::ReadRecordBatch(record_batch->schema(), device_serialized,
                        default_memory_pool(), &read_batch);

  std::cout << read_batch->num_rows() << std::endl;
  std::cout << read_batch->num_columns() << std::endl;

  std::shared_ptr<cuda::CudaIpcMemHandle> handle;

  device_serialized->ExportForIpc(&handle);

  std::shared_ptr<Buffer> buffer;

  handle->Serialize(default_memory_pool(), &buffer);

  std::cout << "here";
}
