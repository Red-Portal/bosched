// PR c++/80485
// { dg-do compile { target c++11 } }
<<<<<<< HEAD
// { dg-options "-fdelete-null-pointer-checks" }
=======
>>>>>>> 3e0e7d8b5b9f61b4341a582fa8c3479ba3b5fdcf

struct dummy {
  void nonnull() {};
  void nonnull2();
};

typedef void (dummy::*safe_bool)();

constexpr safe_bool a = &dummy::nonnull;
constexpr safe_bool b = &dummy::nonnull2;

static_assert( static_cast<bool>( a ), "" );
static_assert( static_cast<bool>( b ), "" );
