// PR c++/85706
<<<<<<< HEAD
// { dg-do compile { target c++17 } }
// { dg-additional-options "-fconcepts" }
=======
// { dg-additional-options "-std=c++17 -fconcepts" }
>>>>>>> 3e0e7d8b5b9f61b4341a582fa8c3479ba3b5fdcf

template<class T> struct S {
  S(T);
};

template<class = void>
auto f() -> decltype(S(42)); // error
