#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>

#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>

#include <arrow/api.h>
#include <arrow/builder.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/api.h>

// Allocating buffer
void allocate_buffer(std::shared_ptr<arrow::Buffer> &buffer)
{
  const int64_t size = 4096;
  if (!arrow::AllocateBuffer(size, &buffer).ok())
  {
    throw std::runtime_error("Buffer allocation was unsuccessful.");
  }
  uint8_t *data = buffer->mutable_data();
  memcpy(data, "Chelsea, hello!", 20);
}

// allocate buffer using builder
template <typename T, typename U>
std::shared_ptr<arrow::Array>
append_arrow_builder(const std::vector<U> &values,
                     const std::vector<bool> &valid)
{
  arrow::NumericBuilder<T> builder;
  std::shared_ptr<arrow::Array> arrow_array;

  builder.Resize(values.size());
  builder.AppendValues(values, valid);

  if (!builder.Finish(&arrow_array).ok())
  {
    throw std::runtime_error("Unable to append Array.");
  }
  return arrow_array;
}

void make_schema(std::shared_ptr<arrow::Schema> &schema)
{
  std::vector<std::shared_ptr<arrow::Field>> fields;

  fields.push_back(arrow::field("int64_", arrow::int64(), true));
  fields.push_back(arrow::field("date32_", arrow::date32(), true));

  schema = arrow::schema(fields);
}

void make_recordbatch(std::shared_ptr<arrow::RecordBatch> &record_batch,
                      std::shared_ptr<arrow::Schema> &schema,
                      std::vector<std::shared_ptr<arrow::Array>> &arrays)
{
  record_batch = arrow::RecordBatch::Make(schema, arrays[0]->length(), arrays);
}

uint8_t *get_and_copy_to_shm(const std::shared_ptr<arrow::Buffer> &data)
{
  if (!data->size())
  {
    throw std::runtime_error("No data to copy.");
  }

  auto key = static_cast<key_t>(rand());
  const auto shmsz = data->size();
  int shmid = -1;

  while ((shmid = shmget(key, shmsz, IPC_CREAT | IPC_EXCL | 0666)) < 0)
  {
    if (!(errno & (EEXIST | EACCES | EINVAL | ENOENT)))
    {
      throw std::runtime_error("failed to create a shared memory.");
    }
    key = static_cast<key_t>(rand());
  }

  auto ipc_ptr = shmat(shmid, NULL, 0);
  if (reinterpret_cast<int64_t>(ipc_ptr) == -1)
  {
    throw std::runtime_error("failed to get shared memory pointer");
  }

  memcpy(ipc_ptr, data->data(), data->size());

  return static_cast<uint8_t *>(ipc_ptr);
}

void print_serialized_schema(const uint8_t *data, const size_t length)
{
  arrow::io::BufferReader reader(std::make_shared<arrow::Buffer>(data, length));
  std::shared_ptr<arrow::Schema> schema;
  arrow::ipc::DictionaryMemo dict_memo;
  if (!arrow::ipc::ReadSchema(&reader, &dict_memo, &schema).ok())
  {
    throw std::runtime_error("Schema not read");
  };

  std::cout << "Arrow Schema: " << std::endl;
  const arrow::PrettyPrintOptions options{0};
  if (!arrow::PrettyPrint(*(schema.get()), options, &std::cout).ok())
  {
    throw std::runtime_error("Unable to print schema.");
  };
  std::cout << std::endl;
}

void print_serialized_records(const uint8_t *data, const size_t length,
                              const std::shared_ptr<arrow::Schema> &schema)
{
  if (data == nullptr || !length)
  {
    std::cout << "No row found" << std::endl;
    return;
  }
  std::shared_ptr<arrow::RecordBatch> batch;
  arrow::ipc::DictionaryMemo dict_memo;

  arrow::io::BufferReader buffer_reader(
      std::make_shared<arrow::Buffer>(data, length));
  if (!arrow::ipc::ReadRecordBatch(schema, &dict_memo, &buffer_reader, &batch).ok())
  {
    throw std::runtime_error("cannot read record batch;");
  }
  std::cout << "Arrow Records: " << std::endl;
  if (!arrow::PrettyPrint(*(batch.get()), 5, &std::cout).ok())
  {
    throw std::runtime_error("Unable to print records");
  }
  std::cout << std::endl;
}

int main()
{
  // test
  std::shared_ptr<arrow::Buffer> buffer;
  allocate_buffer(buffer);
  std::cout << buffer->data() << std::endl;

  //  build arrays
  std::vector<std::shared_ptr<arrow::Array>> arrow_arrays;
  std::vector<int64_t> vec = {13, std::numeric_limits<int64_t>::min() + 1, 87};
  std::vector<bool> valid = {1, 0, 1};
  arrow_arrays.push_back(
      std::move(append_arrow_builder<arrow::Int64Type, int64_t>(vec, valid)));
  std::vector<int32_t> vec1 = {18129, 18128,
                               std::numeric_limits<int32_t>::min() + 1};
  std::vector<bool> valid1 = {1, 1, 0};
  arrow_arrays.push_back(std::move(
      append_arrow_builder<arrow::Date32Type, int32_t>(vec1, valid1)));

  // build schema
  std::shared_ptr<arrow::Schema> schema;
  make_schema(schema);

  // build record batch
  std::shared_ptr<arrow::RecordBatch> record_batch;
  make_recordbatch(record_batch, schema, arrow_arrays);

  // Serialize Schema
  std::shared_ptr<arrow::Buffer> serialized_schema;
  arrow::ipc::DictionaryMemo dict_memo;
  if (!arrow::ipc::SerializeSchema(*record_batch->schema(),
                                   &dict_memo,
                                   arrow::default_memory_pool(),
                                   &serialized_schema)
           .ok())
  {
    throw std::runtime_error("Error: Serializing Schema.");
  }

  // Serialize Record Batch
  std::shared_ptr<arrow::Buffer> serialized_buffer;
  if (!arrow::ipc::SerializeRecordBatch(
           *record_batch, arrow::default_memory_pool(), &serialized_buffer)
           .ok())
  {
    throw std::runtime_error("Error: Serializing Records.");
  }

  // print schema
  const auto schema_ptr = get_and_copy_to_shm(serialized_schema);
  print_serialized_schema(schema_ptr, serialized_schema->size());

  // print records
  const auto records_ptr = get_and_copy_to_shm(serialized_buffer);
  print_serialized_records(records_ptr, serialized_buffer->size(), schema);
}