// PR c++/85305
<<<<<<< HEAD
// { dg-do compile { target c++17 } }
=======
// { dg-additional-options -std=c++17 }
>>>>>>> 3e0e7d8b5b9f61b4341a582fa8c3479ba3b5fdcf

template <int... Is>
void foo()
{
  ([i = Is]{}(), ...); 
}
