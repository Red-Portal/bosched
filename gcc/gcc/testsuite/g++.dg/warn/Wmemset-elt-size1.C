// PR c++/85710
// { dg-additional-options -Wmemset-elt-size }

#include <cstring>

template <typename T> struct A { int a; };

<<<<<<< HEAD
void foo(A<int> (*ap)[2])
{
  std::memset (*ap, 0, 2);	// no warning because A<int> is incomplete
}

=======
>>>>>>> 3e0e7d8b5b9f61b4341a582fa8c3479ba3b5fdcf
template <typename T>
class E
{
public:
  void Clear();
private:
  A<T> mA[2];
};

template<typename T>
void E<T>::Clear()
{
<<<<<<< HEAD
  std::memset(mA, 0, 2);	// { dg-warning -Wmemset-elt-size }
}

int main()
{
  E<int>().Clear();
=======
  std::memset(mA, 0, 2);
>>>>>>> 3e0e7d8b5b9f61b4341a582fa8c3479ba3b5fdcf
}
