// PR c++/85873
<<<<<<< HEAD
// Test that these array temporaries are promoted to static variables as an
=======
// Test that the array temporary is promoted to a static variable as an
>>>>>>> 3e0e7d8b5b9f61b4341a582fa8c3479ba3b5fdcf
// optimization.

// { dg-do compile { target c++11 } }
// { dg-additional-options -fdump-tree-gimple }
// { dg-final { scan-tree-dump-not "= 42" "gimple" } }

#include <initializer_list>

<<<<<<< HEAD
int f()
{
  using AR = const int[];
  return AR{ 1,42,3,4,5,6,7,8,9,0 }[5];
}

=======
>>>>>>> 3e0e7d8b5b9f61b4341a582fa8c3479ba3b5fdcf
int g()
{
  std::initializer_list<int> a = {1,42,3};
  return a.begin()[0];
}
