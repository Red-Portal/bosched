// PR c++/67007
// { dg-options "-std=c++17 -fconcepts" }

template <class U>
concept bool A =
  requires (U u) { u; };

template <class T>
concept bool B =
  requires (T t) { { t } -> A; };

void foo(B);
