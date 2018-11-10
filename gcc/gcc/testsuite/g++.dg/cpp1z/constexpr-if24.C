// PR c++/86480
<<<<<<< HEAD
// { dg-do compile { target c++17 } }
=======
// { dg-additional-options -std=c++17 }
>>>>>>> 3e0e7d8b5b9f61b4341a582fa8c3479ba3b5fdcf

template <class...> constexpr bool val = true;

template <class... T>
void f()
{
  [](auto... p)
    {
<<<<<<< HEAD
      []{
=======
      [=]{
>>>>>>> 3e0e7d8b5b9f61b4341a582fa8c3479ba3b5fdcf
	if constexpr (val<T..., decltype(p)...>) { return true; }
	return false;
      }();
    }(42);
}

int main()
{
  f<int>();
}
