// PR debug/86687
// { dg-do run }
// { dg-options "-g" }

class string {
public:
  string (int p) { this->p = p ; }
  string (const string &s) { this->p = s.p; }

  int p;
};

class foo {
public:
  foo (string dir_hint) {
<<<<<<< HEAD
    p = dir_hint.p; // { dg-final { gdb-test . "dir_hint.p" 3 } }
=======
    p = dir_hint.p; // { dg-final { gdb-test 16 "dir_hint.p" 3 } }
>>>>>>> 3e0e7d8b5b9f61b4341a582fa8c3479ba3b5fdcf
  }

  int p;
};

int
main (void)
{
  string s = 3;
  foo bar(s);
  return !(bar.p == 3);
}
