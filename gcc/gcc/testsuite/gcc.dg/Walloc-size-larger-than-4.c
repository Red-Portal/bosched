/* PR middle-end/82063 - issues with arguments enabled by -Wall
   { dg-do compile }
<<<<<<< HEAD
   { dg-skip-if "small address space" { "pdp11-*-*" } }
=======
>>>>>>> 3e0e7d8b5b9f61b4341a582fa8c3479ba3b5fdcf
   { dg-options "-O -Walloc-size-larger-than=1MiB -ftrack-macro-expansion=0" } */

void sink (void*);

#define T(x) sink (x)

void f (void)
{
  unsigned n = 0;
  T (__builtin_malloc (n));

  n = 1024 * 1024;   /* 1 mebibyte (MiB) */
  T (__builtin_malloc (n));

  n += 1;
  T (__builtin_malloc (n));   /* { dg-warning "argument 1 value .1048577. exceeds maximum object size 1048576" } */
}