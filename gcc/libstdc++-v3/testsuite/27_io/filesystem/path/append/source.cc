// { dg-options "-std=gnu++17 -lstdc++fs" }
// { dg-do run { target c++17 } }
// { dg-require-filesystem-ts "" }

// Copyright (C) 2018 Free Software Foundation, Inc.
//
// This file is part of the GNU ISO C++ Library.  This library is free
// software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the
// Free Software Foundation; either version 3, or (at your option)
// any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License along
// with this library; see the file COPYING3.  If not see
// <http://www.gnu.org/licenses/>.

// C++17 30.10.8.4.3 path appends [fs.path.append]

#include <filesystem>
#include <string_view>
#include <testsuite_fs.h>
#include <testsuite_iterators.h>

using std::filesystem::path;
using __gnu_test::compare_paths;

// path::operator/=(const Source& source)
// path::append(const Source& source)
// Equivalent to: return operator/=(path(source));

// path::append(InputIterator first, InputIterator last)
// Equivalent to: return operator/=(path(first, last));

<<<<<<< HEAD
template<typename Char>
void test(const path& p, const Char* s)
=======
void test(const path& p, std::string_view s)
>>>>>>> 3e0e7d8b5b9f61b4341a582fa8c3479ba3b5fdcf
{
  path expected = p;
  expected /= path(s);

  path oper = p;
  oper /= s;

  path func = p;
  func.append(s);

<<<<<<< HEAD
  __gnu_test::test_container<const Char, __gnu_test::input_iterator_wrapper>
    input_range(s, s + std::char_traits<Char>::length(s));
=======
  __gnu_test::test_container<const char, __gnu_test::input_iterator_wrapper>
    input_range(s.begin(), s.end());
>>>>>>> 3e0e7d8b5b9f61b4341a582fa8c3479ba3b5fdcf
  path range = p;
  range.append(input_range.begin(), input_range.end());

  compare_paths( oper, expected );
  compare_paths( func, expected );
  compare_paths( range, expected );
}

void
test01()
{
  test( "/foo/bar", "/foo/" );

  test( "baz", "baz" );
  test( "baz/", "baz" );
  test( "baz", "/foo/bar" );
  test( "baz/", "/foo/bar" );

  test( "", "" );
  test( "", "rel" );

  test( "dir/", "/file" );
  test( "dir/", "file" );
}

void
test02()
{
  // C++17 [fs.path.append] p4
  test( "//host", "foo" );
  test( "//host/", "foo" );
  test( "foo", "" );
  test( "foo", "/bar" );
  test( "foo", "c:/bar" );
  test( "foo", "c:" );
  test( "c:", "" );
  test( "c:foo", "/bar" );
  test( "foo", "c:\\bar" );
}

void
test03()
{
  for (const path& p : __gnu_test::test_paths)
    for (const path& q : __gnu_test::test_paths)
<<<<<<< HEAD
    {
      test(p, q.c_str());
      if constexpr (!std::is_same_v<path::value_type, char>)
	test(p, q.string().c_str());
    }
}

void
test04()
{
#ifdef _GLIBCXX_USE_WCHAR_T
  test(  "foo", L"/bar" );
  test( L"foo",  "/bar" );
  test( L"foo", L"/bar" );
#endif
=======
      test(p, q.native());
>>>>>>> 3e0e7d8b5b9f61b4341a582fa8c3479ba3b5fdcf
}

int
main()
{
  test01();
  test02();
  test03();
<<<<<<< HEAD
  test04();
=======
>>>>>>> 3e0e7d8b5b9f61b4341a582fa8c3479ba3b5fdcf
}
