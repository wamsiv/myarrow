#include <iostream>
#include <cstdio>

#include <arrow/api.h>
#include <arrow/ipc/api.h>
#include <arrow/io/memory.h>

inline constexpr auto arrow_throw_not_ok = [](auto status) {
    if (!status.ok())
    {
        throw std::runtime_error(status.ToString());
    }
};

inline constexpr auto check = [](auto is_true) {
    if (!is_true)
    {
        throw std::runtime_error("Check Failed.");
    }
};

int main()
{
    std::vector<std::string> list = {"chelsea", "arsenal", "fulham"};

    // initialize schema
    std::shared_ptr<arrow::DataType> dict_type = arrow::dictionary(arrow::int8(), arrow::utf8());
    std::shared_ptr<arrow::Schema> schema = arrow::schema({arrow::field("col1", dict_type, false)});

    // initialize dictionary memo
    arrow::ipc::DictionaryMemo dict_memo;
    std::shared_ptr<arrow::Array> dict;
    check(!dict_memo.HasDictionary(0));
    arrow::StringBuilder builder;
    arrow_throw_not_ok(builder.AppendValues(list));
    arrow_throw_not_ok(builder.Finish(&dict));
    arrow_throw_not_ok(dict_memo.AddDictionary(0, dict));
    check(dict_memo.HasDictionary(0));

    // serialize schema
    std::shared_ptr<arrow::Buffer> serialized_schema;
    arrow_throw_not_ok(arrow::ipc::SerializeSchema(*schema,
                                                   &dict_memo,
                                                   arrow::default_memory_pool(),
                                                   &serialized_schema));

    // read schema
    arrow::io::BufferReader schema_reader(serialized_schema);
    std::shared_ptr<arrow::Schema> deserialized_schema;
    arrow::ipc::DictionaryMemo deserialized_dict_memo;
    arrow_throw_not_ok(arrow::ipc::ReadSchema(&schema_reader, &deserialized_dict_memo, &deserialized_schema));
    std::shared_ptr<arrow::Array> deserialized_dict;
    arrow_throw_not_ok(deserialized_dict_memo.GetDictionary(0, &deserialized_dict));
}