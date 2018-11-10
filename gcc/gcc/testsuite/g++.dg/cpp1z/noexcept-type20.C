// PR c++/85545
// { dg-do compile { target c++11 } }

struct A
{
<<<<<<< HEAD
  constexpr int foo() const noexcept { return 1; }
};

constexpr auto p = static_cast<int (A::*)() const>(&A::foo);
constexpr int i = (A().*p)();

#define SA(X) static_assert((X),#X)
SA(i == 1);
=======
  void foo() noexcept;
};

template<typename T> void bar(T);

void baz()
{
  bar(static_cast<void(A::*)()>(&A::foo));
}
>>>>>>> 3e0e7d8b5b9f61b4341a582fa8c3479ba3b5fdcf
