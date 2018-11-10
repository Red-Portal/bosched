// Copyright (C) 2017-2018 Free Software Foundation, Inc.
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

// { dg-options "-std=gnu++17 -lstdc++fs" }
// { dg-do run { target c++17 } }
// { dg-require-filesystem-ts "" }

// 30.10.14.3 Permissions [fs.op.space]

#include <filesystem>
#include <testsuite_fs.h>
#include <testsuite_hooks.h>

bool check(std::filesystem::space_info const& s)
{
  const std::uintmax_t err = -1;
  return s.capacity != err || s.free != err || s.available != err;
}

void
test01()
{
  const std::filesystem::path root = __gnu_test::root_path();
  std::filesystem::space_info s = std::filesystem::space(root);
  std::error_code ec = make_error_code(std::errc::invalid_argument);
  s = std::filesystem::space(root, ec);
  VERIFY( !ec );
<<<<<<< HEAD
  VERIFY( check(s) );
  VERIFY( s.capacity >= s.free );

  s = std::filesystem::space(__gnu_test::nonexistent_path()/".", ec);
  if (ec)
    VERIFY( ! check(s) );
  else
    VERIFY( check(s) );
}

void
test02()
{
  std::filesystem::space_info s = std::filesystem::space(".");
  VERIFY( check(s) );
  VERIFY( s.capacity >= s.free );
=======

  s = std::filesystem::space(__gnu_test::nonexistent_path(), ec);
  VERIFY( ec );
  VERIFY( s.capacity ==  static_cast<uintmax_t>(-1) );
  VERIFY( s.free ==  static_cast<uintmax_t>(-1) );
  VERIFY( s.available ==  static_cast<uintmax_t>(-1) );
>>>>>>> 3e0e7d8b5b9f61b4341a582fa8c3479ba3b5fdcf
}

void
test02()
{
  std::filesystem::space_info s = std::filesystem::space(".");
  VERIFY( s.capacity >= s.free );
}

int
main()
{
  test01();
  test02();
}