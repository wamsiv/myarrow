#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>

#include <arrow/api.h>
#include <arrow/ipc/api.h>
#include <arrow/builder.h>
#include <arrow/io/memory.h>

__global__ void initialize_array(const int64_t sz, int64_t *arr) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;

  for (int64_t i = idx; i < sz; i += stride) {
    arr[i] = i+1;
  }
}

__global__ void print_array(const int64_t sz, int64_t *arr) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;

  for (int64_t i = idx; i < sz; i += stride) {
    printf("%d,", arr[i]);
  }
}

void make_schema(std::shared_ptr<arrow::Schema> schema) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    fields.push_back(arrow::field("int_", arrow::int64(), false));
    schema = arrow::schema(fields);
}

//get record batch
void make_record_batch(std::shared_ptr<arrow::RecordBatch>& record_batch, 
                        std::shared_ptr<arrow::Schema>& schema, int64_t* arr,
                        const int64_t sz) {
    arrow::Int64Builder builder;
    std::shared_ptr<arrow::Array> arrow_array;
    std::vector<bool> valid = {1,1,1,1};
    builder.Resize(sz);
    builder.AppendValues(arr, sz, valid);
    if (!builder.Finish(&arrow_array).ok()) {
        throw std::runtime_error("Unable to append array.");
    }
    std::vector<std::shared_ptr<arrow::Array>> arrow_arrays;
    arrow_arrays.push_back(arrow_array);
    record_batch = arrow::RecordBatch::Make(schema, arrow_arrays[0]->length(), arrow_arrays);
}

uint8_t *get_and_copy_to_shm(const std::shared_ptr<arrow::Buffer> &data) {
    if (!data->size()) {
      throw std::runtime_error("No data to copy.");
    }
  
    auto key = static_cast<key_t>(rand());
    const auto shmsz = data->size();
    int shmid = -1;
  
    while ((shmid = shmget(key, shmsz, IPC_CREAT | IPC_EXCL | 0666)) < 0) {
      if (!(errno & (EEXIST | EACCES | EINVAL | ENOENT))) {
        throw std::runtime_error("failed to create a shared memory.");
      }
      key = static_cast<key_t>(rand());
    }
  
    auto ipc_ptr = shmat(shmid, NULL, 0);
    if (reinterpret_cast<int64_t>(ipc_ptr) == -1) {
      throw std::runtime_error("failed to get shared memory pointer");
    }
  
    memcpy(ipc_ptr, data->data(), data->size());
  
    return static_cast<uint8_t *>(ipc_ptr);
  }

void print_serialized_records(const uint8_t *data, const size_t length,
    const std::shared_ptr<arrow::Schema> &schema) {
if (data == nullptr || !length) {
std::cout << "No row found" << std::endl;
return;
}
std::shared_ptr<arrow::RecordBatch> batch;

arrow::io::BufferReader buffer_reader(
std::make_shared<arrow::Buffer>(data, length));
if (!arrow::ipc::ReadRecordBatch(schema, &buffer_reader, &batch).ok()) {
throw std::runtime_error("cannot read record batch;");
}
std::cout << "Arrow Records: " << std::endl;
if (!arrow::PrettyPrint(*(batch.get()), 5, &std::cout).ok()) {
throw std::runtime_error("Unable to print records");
}
std::cout << std::endl;
}

int main() {
  const int64_t sz = 1 << 2;
  int64_t *devarr;
//   cudaMallocManaged(&devarr, sz * sizeof(int64_t));
  cudaMalloc((int64_t**)&devarr, sz*sizeof(int64_t));
  int64_t blocksize = 256;
  int64_t numblocks = (sz + blocksize - 1) / blocksize;
  initialize_array<<<numblocks, blocksize>>>(sz, devarr);
  cudaDeviceSynchronize();
  print_array<<<numblocks, blocksize>>>(sz, devarr);
  cudaDeviceSynchronize();
  std::shared_ptr<arrow::Schema> schema;
  std::shared_ptr<arrow::RecordBatch> batch;
  make_schema(schema);
  make_record_batch(batch, schema, devarr, sz);
  // Serialize Record Batch
  std::shared_ptr<arrow::Buffer> serialized_buffer;
  if (!arrow::ipc::SerializeRecordBatch(
           *batch, arrow::default_memory_pool(), &serialized_buffer).ok()) {
    throw std::runtime_error("Error: Serializing Records.");
  }
  
  const auto records_ptr = get_and_copy_to_shm(serialized_buffer);
  print_serialized_records(records_ptr, serialized_buffer->size(), schema);
  
  // free memeory
  cudaFree(devarr);
  
}