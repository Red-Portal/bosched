// { dg-do compile }
// { dg-additional-options "-flto" { target lto } }
struct X {
<<<<<<< HEAD
  int *__attribute__((aligned(2), packed)) a; // { dg-warning "ignoring attribute .packed. because it conflicts with attribute .aligned." }
=======
  int *__attribute__((aligned(2), packed)) a; // { dg-warning "attribute ignored" }
>>>>>>> 3e0e7d8b5b9f61b4341a582fa8c3479ba3b5fdcf
} b;
