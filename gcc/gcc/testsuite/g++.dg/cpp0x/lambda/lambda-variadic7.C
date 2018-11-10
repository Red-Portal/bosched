// PR c++/86200
// { dg-do compile { target c++11 } }

template<typename ... Args>
static void foo()
{
<<<<<<< HEAD
  [](Args, int x) {		// { dg-error "packs not expanded" }
    x;
  };
=======
  [](Args, int x) {
    x;
  };				// { dg-error "packs not expanded" }
>>>>>>> 3e0e7d8b5b9f61b4341a582fa8c3479ba3b5fdcf
}
int main()
{
  foo();
}
