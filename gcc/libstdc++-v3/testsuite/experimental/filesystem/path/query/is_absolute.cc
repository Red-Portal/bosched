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

// 8.4.9 path decomposition [path.decompose]

#include <filesystem>
#include <testsuite_hooks.h>

using std::filesystem::path;

void
test01()
{
<<<<<<< HEAD
  #ifdef _GLIBCXX_FILESYSTEM_IS_WINDOWS
  const bool is_posix = false;
#else
  const bool is_posix = true;
#endif

  VERIFY( path("/").is_absolute() == is_posix );
  VERIFY( path("/foo").is_absolute() == is_posix );
  VERIFY( path("/foo/").is_absolute() == is_posix );
  VERIFY( path("/foo/bar").is_absolute() == is_posix );
  VERIFY( path("/foo/bar/").is_absolute() == is_posix );
=======
  VERIFY( path("/").is_absolute() );
  VERIFY( path("/foo").is_absolute() );
  VERIFY( path("/foo/").is_absolute() );
  VERIFY( path("/foo/bar").is_absolute() );
  VERIFY( path("/foo/bar/").is_absolute() );
>>>>>>> 3e0e7d8b5b9f61b4341a582fa8c3479ba3b5fdcf
  VERIFY( ! path("foo").is_absolute() );
  VERIFY( ! path("foo/").is_absolute() );
  VERIFY( ! path("foo/bar").is_absolute() );
  VERIFY( ! path("foo/bar/").is_absolute() );
  VERIFY( ! path("c:").is_absolute() );
  VERIFY( ! path("c:foo").is_absolute() );
  VERIFY( ! path("c:foo/").is_absolute() );
  VERIFY( ! path("c:foo/bar").is_absolute() );
  VERIFY( ! path("c:foo/bar/").is_absolute() );
<<<<<<< HEAD
  VERIFY( path("c:/").is_absolute() == !is_posix );
  VERIFY( path("c:/foo").is_absolute() == !is_posix );
  VERIFY( path("c:/foo/").is_absolute() == !is_posix );
  VERIFY( path("c:/foo/bar").is_absolute() == !is_posix );
  VERIFY( path("c:/foo/bar/").is_absolute() == !is_posix );
=======
#ifdef _GLIBCXX_FILESYSTEM_IS_WINDOWS
  const bool drive_letter_is_root_name = true;
#else
  const bool drive_letter_is_root_name = false;
#endif
  VERIFY( path("c:/").is_absolute() == drive_letter_is_root_name );
  VERIFY( path("c:/foo").is_absolute() == drive_letter_is_root_name );
  VERIFY( path("c:/foo/").is_absolute() == drive_letter_is_root_name );
  VERIFY( path("c:/foo/bar").is_absolute() == drive_letter_is_root_name );
  VERIFY( path("c:/foo/bar/").is_absolute() == drive_letter_is_root_name );
>>>>>>> 3e0e7d8b5b9f61b4341a582fa8c3479ba3b5fdcf
}

int
main()
{
  test01();
}
