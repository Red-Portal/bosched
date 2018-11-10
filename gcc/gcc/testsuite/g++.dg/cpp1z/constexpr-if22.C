// PR c++/85695
<<<<<<< HEAD
// { dg-do compile { target c++17 } }
=======
// { dg-options -std=c++17 }
>>>>>>> 3e0e7d8b5b9f61b4341a582fa8c3479ba3b5fdcf

template <typename T, T v>
struct integral_constant {
    using value_type = T;
    static constexpr const value_type value = v;
    constexpr operator value_type (void) const { return value; }
};
template <typename T> struct is_trivial
    : public integral_constant<bool, __is_trivial(T)> {};

template <typename T>
T clone_object (const T& p)
{
    if constexpr (is_trivial<T>::value)
        return p;
    else
        return p.clone();
}
int main (void) { return clone_object(0); }
