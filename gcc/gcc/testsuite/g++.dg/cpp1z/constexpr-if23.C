// PR c++/85842
<<<<<<< HEAD
// { dg-do compile { target c++17 } }
=======
// { dg-additional-options -std=c++17 }
>>>>>>> 3e0e7d8b5b9f61b4341a582fa8c3479ba3b5fdcf

template<class T>
auto f = [](auto&& arg) -> T* {
    if constexpr (sizeof(arg) == 1) {
        return nullptr;
    } else {
        return static_cast<T*>(&arg);
    }
};

auto p = f<int>(0);
