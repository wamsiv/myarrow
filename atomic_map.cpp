#include <iostream>
#include <cstdio>
#include <atomic>
#include <string>
#include <map>

struct User{
    std::atomic<bool> is_super;

    User(){};
    User(const bool super) : is_super(super) {}
};

std::map<std::string, User> map_;


int main() {
    User user;
    user.is_super.store(false, std::memory_order_relaxed);

    // auto emplace_val = map_.emplace(std::string("chelsea"), user);
    map_.insert(std::pair<std::string, User>("chelsea",user));

    for (const auto& it: map_) {
        std::cout << it.first << ", " << it.second.is_super << std::endl;
    }

}
