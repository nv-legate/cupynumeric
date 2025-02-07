/* Copyright 2024 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <iomanip>
#include <iostream>
#include <sstream>

#include <gtest/gtest.h>
#include "legate.h"
#include "cupynumeric.h"
#include "util.inl"

void swapaxes_test()
{
  // Test small
  {
    auto A = cupynumeric::zeros({3, 3}, legate::int32());
    EXPECT_EQ(A.shape(), (std::vector<uint64_t>{3, 3}));
    auto B = cupynumeric::swapaxes(A, 0, 1);
    EXPECT_EQ(B.shape(), (std::vector<uint64_t>{3, 3}));
  }

  // Test tall
  {
    auto A_tall = cupynumeric::zeros({300, 3}, legate::int32());
    EXPECT_EQ(A_tall.shape(), (std::vector<uint64_t>{300, 3}));
    auto B_tall = cupynumeric::swapaxes(A_tall, 0, 1);
    EXPECT_EQ(B_tall.shape(), (std::vector<uint64_t>{3, 300}));
  }

  // Test wide
  {
    auto A_wide = cupynumeric::zeros({3, 300}, legate::int32());
    EXPECT_EQ(A_wide.shape(), (std::vector<uint64_t>{3, 300}));
    auto B_wide = cupynumeric::swapaxes(A_wide, 0, 1);
    EXPECT_EQ(B_wide.shape(), (std::vector<uint64_t>{300, 3}));
  }

  // Test big
  {
    auto A_big = cupynumeric::zeros({300, 300}, legate::int32());
    EXPECT_EQ(A_big.shape(), (std::vector<uint64_t>{300, 300}));
    auto B_big = cupynumeric::swapaxes(A_big, 0, 1);
    EXPECT_EQ(B_big.shape(), (std::vector<uint64_t>{300, 300}));
  }

  // Test 3-dim array with different swap axes
  {
    auto A = cupynumeric::zeros({3, 4, 5}, legate::int32());
    EXPECT_EQ(A.shape(), (std::vector<uint64_t>{3, 4, 5}));

    auto B1 = cupynumeric::swapaxes(A, 0, 0);
    EXPECT_EQ(B1.shape(), (std::vector<uint64_t>{3, 4, 5}));

    auto B2 = cupynumeric::swapaxes(A, -3, 1);
    EXPECT_EQ(B2.shape(), (std::vector<uint64_t>{4, 3, 5}));

    auto B3 = cupynumeric::swapaxes(A, 0, 2);
    EXPECT_EQ(B3.shape(), (std::vector<uint64_t>{5, 4, 3}));

    auto B4 = cupynumeric::swapaxes(A, -3, -2);
    EXPECT_EQ(B4.shape(), (std::vector<uint64_t>{4, 3, 5}));
  }

  // Test empty array
  {
    auto A = cupynumeric::zeros({0}, legate::int32());
    EXPECT_EQ(A.shape(), (std::vector<uint64_t>{0}));

    auto B = cupynumeric::swapaxes(A, 0, 0);
    EXPECT_EQ(B.shape(), (std::vector<uint64_t>{0}));
  }
}

void swapaxes_negative_test()
{
  // Test out-of-bound1
  auto A = cupynumeric::zeros({3, 3}, legate::int32());
  EXPECT_THROW(cupynumeric::swapaxes(A, 3, 0), std::invalid_argument);
  EXPECT_THROW(cupynumeric::swapaxes(A, 0, 3), std::invalid_argument);

  // Test out-of-bound2
  EXPECT_THROW(cupynumeric::swapaxes(A, -4, 0), std::invalid_argument);
  EXPECT_THROW(cupynumeric::swapaxes(A, 0, -4), std::invalid_argument);
}

// void cpp_test()
TEST(Swapaxes, Normal) { swapaxes_test(); }

TEST(Swapaxes, Negative) { swapaxes_negative_test(); }
