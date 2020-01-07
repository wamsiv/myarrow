#include <arrow/api.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/api.h>
#include <cstdio>
#include <iostream>

int main() {
  std::shared_ptr<arrow::Schema> schema =
      arrow::schema({arrow::field("int_", arrow::int32(), false)});
  std::vector<std::shared_ptr<arrow::Array>> arrays = {};

  std::shared_ptr<arrow::RecordBatch> record_batch =
      arrow::RecordBatch::Make(schema, arrays[0]->length(), arrays);
  std::shared_ptr<arrow::Buffer> serialized_buffer;
  if (!arrow::ipc::SerializeRecordBatch(
           *record_batch, arrow::default_memory_pool(), &serialized_buffer)
           .ok()) {
    throw std::runtime_error("Error: Serializing Records.");
  }
}