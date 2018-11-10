/* PR middle-end/82063 - issues with arguments enabled by -Wall
<<<<<<< HEAD
   Verify that an invalid argument to -Walloc-size-larger-than is diagnosed.
   { dg-do compile }
   { dg-options "-Walloc-size-larger-than=1zb -Walloca-larger-than=2kbytes -Wvla-larger-than=3MIBZ" } */


/* { dg-error "argument to '-Walloc-size-larger-than=' should be a non-negative integer optionally followed by a size unit" "" { target *-*-* } 0 }
   { dg-error "argument to '-Walloca-larger-than=' should be a non-negative integer optionally followed by a size unit" "" { target *-*-* } 0 }
   { dg-error "argument to '-Wvla-larger-than=' should be a non-negative integer optionally followed by a size unit" "" { target *-*-* } 0 } */
=======
   { dg-do compile }
   { dg-options "-O -Walloc-size-larger-than=1zb -ftrack-macro-expansion=0" } */

typedef __SIZE_TYPE__ size_t;

void sink (void*);

#define T(x) sink (x)

/* Verify that an invalid -Walloc-size-larger-than argument is diagnosed
   and rejected without changing the default setting of PTRDIFF_MAX.  */

void f (void)
{
  size_t n = 0;
  T (__builtin_malloc (n));

  n = __PTRDIFF_MAX__;
  T (__builtin_malloc (n));

  n += 1;
  T (__builtin_malloc (n));   /* { dg-warning "exceeds maximum object size" } */

  n = __SIZE_MAX__ - 1;
  T (__builtin_malloc (n));   /* { dg-warning "exceeds maximum object size" } */

  n = __SIZE_MAX__;
  T (__builtin_malloc (n));   /* { dg-warning "exceeds maximum object size" } */
}

/* { dg-warning "invalid argument .1zb. to .-Walloc-size-larger-than=." "" { target *-*-* } 0 } */
>>>>>>> 3e0e7d8b5b9f61b4341a582fa8c3479ba3b5fdcf
