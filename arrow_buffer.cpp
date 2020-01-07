#include <cstdio>
#include <climits>
#include <cstdio>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>

#include <arrow/io/memory.h>
#include <arrow/api.h>
#include <arrow/ipc/api.h>

void allocate_array_buffer(std::shared_ptr<arrow::Buffer>& buffer) {
    const int64_t sz = 5;
    int64_t* arr = new int64_t[sz];
    for (int i=0; i < sz; ++i) {
        arr[i] = i+1;
    }
    if (!arrow::AllocateBuffer(sz*sizeof(arr), &buffer).ok()) {
        throw std::runtime_error("Buffer allocation not successful!");
    }
    uint8_t* write = buffer->mutable_data();
    memcpy(write, arr, sz*sizeof(arr));
    free(arr);
}

void get_array_schema(std::shared_ptr<arrow::Schema>& schema) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    fields.push_back(arrow::field("int_", arrow::int32(),false));
    schema = arrow::schema(fields);
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

void print_serialized_schema(const uint8_t *data, const size_t length) {
  arrow::io::BufferReader reader(std::make_shared<arrow::Buffer>(data, length));
  std::shared_ptr<arrow::Schema> schema;
  if (!arrow::ipc::ReadSchema(&reader, &schema).ok()) {
    throw std::runtime_error("Schema not read");
  };

  std::cout << "Arrow Schema: " << std::endl;
  const arrow::PrettyPrintOptions options{0};
  if (!arrow::PrettyPrint(*(schema.get()), options, &std::cout).ok()) {
    throw std::runtime_error("Unable to print schema.");
  };
  std::cout << std::endl;
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
    std::shared_ptr<arrow::Buffer> buffer;
    allocate_array_buffer(buffer);
    // std::cout << buffer->data() << std::endl;
    std::shared_ptr<arrow::Schema> schema;
    get_array_schema(schema);

    const auto buffer_ptr = get_and_copy_to_shm(buffer);
    print_serialized_records(buffer_ptr, buffer->size(), schema);
}