
#include <cstdio>
#include <iostream>

#include <type_traits>
#include <vector>

#include <optional>

template <typename... PRIMITIVE_TYPES>
class OptionalVecAmalgamation
    : public std::optional<std::vector<PRIMITIVE_TYPES>>... {
public:
  template <typename T> const auto &getBaseVec() { return getBaseVecImpl<T>(); }

  template <typename T> auto &getMutableBaseVec() {
    return getBaseVecImpl<T>();
  }

  const auto empty() { return isEmpty(); }

private:
  template <typename T> auto &getBaseVecImpl() {
    return static_cast<std::optional<std::vector<T>> &>(*this);
  }

  const auto isEmpty() {
    return (static_cast<std::optional<std::vector<PRIMITIVE_TYPES>> &>(*this)
                ->empty() &&
            ...);
  }
};

int main() {
  OptionalVecAmalgamation<int, float, std::string> ot;
  OptionalVecAmalgamation<int, float, std::string> ot1;

  auto &opt_vector_of_interest = ot.getMutableBaseVec<int>();
  opt_vector_of_interest->push_back(1);
  opt_vector_of_interest->push_back(2);

  auto &float_vector_of_interest = ot.getMutableBaseVec<float>();
  float_vector_of_interest->push_back(3.1f);
  float_vector_of_interest->push_back(4.1f);

  for (const auto num : *opt_vector_of_interest) {
    std::cout << num << std::endl;
  }

  for (const auto fnum : *float_vector_of_interest) {
    std::cout << fnum << std::endl;
  }

  for (const auto fnum : *ot.getBaseVec<float>()) {
    std::cout << fnum << std::endl;
  }

  std::cout << ot.empty() << std ::endl;
  std::cout << ot1.empty() << std ::endl;
}
