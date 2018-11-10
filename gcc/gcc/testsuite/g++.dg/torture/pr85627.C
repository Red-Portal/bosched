/* { dg-do compile } */
/* { dg-skip-if "" { *-*-* } { "-flto" } { "" } } */
/* { dg-additional-options "-fnon-call-exceptions -fdump-tree-optimized" } */

__complex double
foo (__complex double a, __complex double b)
{
  __complex res = a;
  try {
      res = a * b;
  }
  catch (...) {
      res = b;
  }
  return res;
}

__complex double
bar (__complex double a, __complex double b)
{
  __complex res = a;
  try {
      res = a / b;
  }
  catch (...) {
      res = b;
  }
  return res;
}
<<<<<<< HEAD

/* Verify EH is preserved by complex lowering.  */

/* { dg-final { scan-tree-dump-times "__cxa_begin_catch" 2 "optimized" } } */
=======
>>>>>>> 3e0e7d8b5b9f61b4341a582fa8c3479ba3b5fdcf
