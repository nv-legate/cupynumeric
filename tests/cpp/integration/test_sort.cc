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

auto get_expect_result_int()
{
  std::vector<std::map<int32_t, std::array<int32_t, 12>>> expect_result = {
    {{0, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}},
    {{-1, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
     {0, {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1}},
     {1, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}},
    {{-1, {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1}},
     {0, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
     {1, {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1}}},
    {{-1, {3, 5, 10, 12, 2, 4, 8, 9, 1, 6, 7, 11}},
     {0, {2, 3, 8, 1, 7, 4, 11, 5, 10, 6, 12, 9}},
     {1, {3, 5, 10, 12, 2, 4, 8, 9, 1, 6, 7, 11}}},
    {{-2, {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1}},
     {-1, {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1}},
     {0, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
     {1, {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1}},
     {2, {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1}}},
    {{-2, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
     {-1, {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1}},
     {0, {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1}},
     {1, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
     {2, {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1}}},
    {{-2, {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1}},
     {-1, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
     {0, {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1}},
     {1, {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1}},
     {2, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}},
    {{-2, {5, 2, 4, 10, 3, 12, 6, 9, 1, 8, 11, 7}},
     {-1, {3, 10, 12, 2, 4, 5, 7, 8, 9, 1, 6, 11}},
     {0, {8, 3, 7, 5, 2, 1, 10, 9, 12, 6, 11, 4}},
     {1, {5, 2, 4, 10, 3, 12, 6, 9, 1, 8, 11, 7}},
     {2, {3, 10, 12, 2, 4, 5, 7, 8, 9, 1, 6, 11}}}};
  return expect_result;
}

auto get_expect_result_double()
{
  std::vector<std::map<int32_t, std::array<double, 12>>> expect_result = {
    {{0, {1.5, 2.2, 3.66, 4, 5.98, 6, 7.9, 8, 9, 10.5, 11, 12}}},
    {{-1, {1.5, 2.2, 3.66, 4, 5.98, 6, 7.9, 8, 9, 10.5, 11, 12}},
     {0, {1.5, 3.66, 6, 5.98, 2.2, 10.5, 8, 11, 7.9, 12, 9, 4}},
     {1, {1.5, 2.2, 3.66, 4, 5.98, 6, 7.9, 8, 9, 10.5, 11, 12}}},
    {{-1, {1.5, 3.66, 6, 5.98, 2.2, 10.5, 8, 11, 7.9, 12, 9, 4}},
     {0, {1.5, 2.2, 3.66, 4, 5.98, 6, 7.9, 8, 9, 10.5, 11, 12}},
     {1, {1.5, 3.66, 6, 5.98, 2.2, 10.5, 8, 11, 7.9, 12, 9, 4}}},
    {{-1, {1.5, 3.66, 5.98, 6, 2.2, 8, 10.5, 11, 4, 7.9, 9, 12}},
     {0, {1.5, 3.66, 6, 4, 2.2, 10.5, 8, 5.98, 7.9, 12, 9, 11}},
     {1, {1.5, 3.66, 5.98, 6, 2.2, 8, 10.5, 11, 4, 7.9, 9, 12}}},
    {{-2, {1.5, 3.66, 6, 5.98, 2.2, 10.5, 8, 11, 7.9, 12, 9, 4}},
     {-1, {1.5, 3.66, 6, 5.98, 2.2, 10.5, 8, 11, 7.9, 12, 9, 4}},
     {0, {1.5, 2.2, 3.66, 4, 5.98, 6, 7.9, 8, 9, 10.5, 11, 12}},
     {1, {1.5, 3.66, 6, 5.98, 2.2, 10.5, 8, 11, 7.9, 12, 9, 4}},
     {2, {1.5, 3.66, 6, 5.98, 2.2, 10.5, 8, 11, 7.9, 12, 9, 4}}},
    {{-2, {1.5, 2.2, 3.66, 4, 5.98, 6, 7.9, 8, 9, 10.5, 11, 12}},
     {-1, {1.5, 3.66, 6, 5.98, 2.2, 10.5, 8, 11, 7.9, 12, 9, 4}},
     {0, {1.5, 3.66, 6, 5.98, 2.2, 10.5, 8, 11, 7.9, 12, 9, 4}},
     {1, {1.5, 2.2, 3.66, 4, 5.98, 6, 7.9, 8, 9, 10.5, 11, 12}},
     {2, {1.5, 3.66, 6, 5.98, 2.2, 10.5, 8, 11, 7.9, 12, 9, 4}}},
    {{-2, {1.5, 3.66, 6, 5.98, 2.2, 10.5, 8, 11, 7.9, 12, 9, 4}},
     {-1, {1.5, 2.2, 3.66, 4, 5.98, 6, 7.9, 8, 9, 10.5, 11, 12}},
     {0, {1.5, 3.66, 6, 5.98, 2.2, 10.5, 8, 11, 7.9, 12, 9, 4}},
     {1, {1.5, 3.66, 6, 5.98, 2.2, 10.5, 8, 11, 7.9, 12, 9, 4}},
     {2, {1.5, 2.2, 3.66, 4, 5.98, 6, 7.9, 8, 9, 10.5, 11, 12}}},
    {{-2, {1.5, 2.2, 6, 5.98, 3.66, 10.5, 8, 9, 4, 12, 11, 7.9}},
     {-1, {1.5, 3.66, 6, 2.2, 5.98, 10.5, 7.9, 8, 11, 4, 9, 12}},
     {0, {1.5, 3.66, 6, 5.98, 2.2, 4, 8, 11, 7.9, 12, 9, 10.5}},
     {1, {1.5, 2.2, 6, 5.98, 3.66, 10.5, 8, 9, 4, 12, 11, 7.9}},
     {2, {1.5, 3.66, 6, 2.2, 5.98, 10.5, 7.9, 8, 11, 4, 9, 12}}}};
  return expect_result;
}

auto get_expect_result_complex()
{
  std::vector<std::map<int32_t, std::array<legate::Complex<float>, 12>>> expect_result = {
    {{0,
      {legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(2, 4),
       legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(6, 4),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(8, 9),
       legate::Complex<float>(8, 11),
       legate::Complex<float>(10, 3),
       legate::Complex<float>(11, 1),
       legate::Complex<float>(12, 5)}}},
    {{-1,
      {legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(2, 4),
       legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(6, 4),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(8, 9),
       legate::Complex<float>(8, 11),
       legate::Complex<float>(10, 3),
       legate::Complex<float>(11, 1),
       legate::Complex<float>(12, 5)}},
     {0,
      {legate::Complex<float>(10, 3),
       legate::Complex<float>(12, 5),
       legate::Complex<float>(2, 4),
       legate::Complex<float>(8, 9),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(11, 1),
       legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(8, 11),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(6, 4)}},
     {1,
      {legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(2, 4),
       legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(6, 4),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(8, 9),
       legate::Complex<float>(8, 11),
       legate::Complex<float>(10, 3),
       legate::Complex<float>(11, 1),
       legate::Complex<float>(12, 5)}}},
    {{-1,
      {legate::Complex<float>(10, 3),
       legate::Complex<float>(12, 5),
       legate::Complex<float>(2, 4),
       legate::Complex<float>(8, 9),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(11, 1),
       legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(8, 11),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(6, 4)}},
     {0,
      {legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(2, 4),
       legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(6, 4),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(8, 9),
       legate::Complex<float>(8, 11),
       legate::Complex<float>(10, 3),
       legate::Complex<float>(11, 1),
       legate::Complex<float>(12, 5)}},
     {1,
      {legate::Complex<float>(10, 3),
       legate::Complex<float>(12, 5),
       legate::Complex<float>(2, 4),
       legate::Complex<float>(8, 9),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(11, 1),
       legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(8, 11),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(6, 4)}}},
    {{-1,
      {legate::Complex<float>(2, 4),
       legate::Complex<float>(8, 9),
       legate::Complex<float>(10, 3),
       legate::Complex<float>(12, 5),
       legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(11, 1),
       legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(6, 4),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(8, 11)}},
     {0,
      {legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(8, 11),
       legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(6, 4),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(11, 1),
       legate::Complex<float>(2, 4),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(10, 3),
       legate::Complex<float>(12, 5),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(8, 9)}},
     {1,
      {legate::Complex<float>(2, 4),
       legate::Complex<float>(8, 9),
       legate::Complex<float>(10, 3),
       legate::Complex<float>(12, 5),
       legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(11, 1),
       legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(6, 4),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(8, 11)}}},
    {{-2,
      {legate::Complex<float>(10, 3),
       legate::Complex<float>(12, 5),
       legate::Complex<float>(2, 4),
       legate::Complex<float>(8, 9),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(11, 1),
       legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(8, 11),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(6, 4)}},
     {-1,
      {legate::Complex<float>(10, 3),
       legate::Complex<float>(12, 5),
       legate::Complex<float>(2, 4),
       legate::Complex<float>(8, 9),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(11, 1),
       legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(8, 11),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(6, 4)}},
     {0,
      {legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(2, 4),
       legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(6, 4),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(8, 9),
       legate::Complex<float>(8, 11),
       legate::Complex<float>(10, 3),
       legate::Complex<float>(11, 1),
       legate::Complex<float>(12, 5)}},
     {1,
      {legate::Complex<float>(10, 3),
       legate::Complex<float>(12, 5),
       legate::Complex<float>(2, 4),
       legate::Complex<float>(8, 9),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(11, 1),
       legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(8, 11),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(6, 4)}},
     {2,
      {legate::Complex<float>(10, 3),
       legate::Complex<float>(12, 5),
       legate::Complex<float>(2, 4),
       legate::Complex<float>(8, 9),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(11, 1),
       legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(8, 11),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(6, 4)}}},
    {{-2,
      {legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(2, 4),
       legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(6, 4),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(8, 9),
       legate::Complex<float>(8, 11),
       legate::Complex<float>(10, 3),
       legate::Complex<float>(11, 1),
       legate::Complex<float>(12, 5)}},
     {-1,
      {legate::Complex<float>(10, 3),
       legate::Complex<float>(12, 5),
       legate::Complex<float>(2, 4),
       legate::Complex<float>(8, 9),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(11, 1),
       legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(8, 11),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(6, 4)}},
     {0,
      {legate::Complex<float>(10, 3),
       legate::Complex<float>(12, 5),
       legate::Complex<float>(2, 4),
       legate::Complex<float>(8, 9),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(11, 1),
       legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(8, 11),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(6, 4)}},
     {1,
      {legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(2, 4),
       legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(6, 4),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(8, 9),
       legate::Complex<float>(8, 11),
       legate::Complex<float>(10, 3),
       legate::Complex<float>(11, 1),
       legate::Complex<float>(12, 5)}},
     {2,
      {legate::Complex<float>(10, 3),
       legate::Complex<float>(12, 5),
       legate::Complex<float>(2, 4),
       legate::Complex<float>(8, 9),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(11, 1),
       legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(8, 11),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(6, 4)}}},
    {{-2,
      {legate::Complex<float>(10, 3),
       legate::Complex<float>(12, 5),
       legate::Complex<float>(2, 4),
       legate::Complex<float>(8, 9),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(11, 1),
       legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(8, 11),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(6, 4)}},
     {-1,
      {legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(2, 4),
       legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(6, 4),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(8, 9),
       legate::Complex<float>(8, 11),
       legate::Complex<float>(10, 3),
       legate::Complex<float>(11, 1),
       legate::Complex<float>(12, 5)}},
     {0,
      {legate::Complex<float>(10, 3),
       legate::Complex<float>(12, 5),
       legate::Complex<float>(2, 4),
       legate::Complex<float>(8, 9),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(11, 1),
       legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(8, 11),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(6, 4)}},
     {1,
      {legate::Complex<float>(10, 3),
       legate::Complex<float>(12, 5),
       legate::Complex<float>(2, 4),
       legate::Complex<float>(8, 9),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(11, 1),
       legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(8, 11),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(6, 4)}},
     {2,
      {legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(2, 4),
       legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(6, 4),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(8, 9),
       legate::Complex<float>(8, 11),
       legate::Complex<float>(10, 3),
       legate::Complex<float>(11, 1),
       legate::Complex<float>(12, 5)}}},
    {{-2,
      {legate::Complex<float>(8, 9),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(2, 4),
       legate::Complex<float>(10, 3),
       legate::Complex<float>(12, 5),
       legate::Complex<float>(11, 1),
       legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(8, 11),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(6, 4)}},
     {-1,
      {legate::Complex<float>(2, 4),
       legate::Complex<float>(10, 3),
       legate::Complex<float>(12, 5),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(8, 9),
       legate::Complex<float>(11, 1),
       legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(6, 4),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(8, 11)}},
     {0,
      {legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(2, 4),
       legate::Complex<float>(8, 9),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(6, 4),
       legate::Complex<float>(10, 3),
       legate::Complex<float>(12, 5),
       legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(8, 11),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(11, 1)}},
     {1,
      {legate::Complex<float>(8, 9),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(2, 4),
       legate::Complex<float>(10, 3),
       legate::Complex<float>(12, 5),
       legate::Complex<float>(11, 1),
       legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(8, 11),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(6, 4)}},
     {2,
      {legate::Complex<float>(2, 4),
       legate::Complex<float>(10, 3),
       legate::Complex<float>(12, 5),
       legate::Complex<float>(7, 6),
       legate::Complex<float>(8, 9),
       legate::Complex<float>(11, 1),
       legate::Complex<float>(1.5, 3.66),
       legate::Complex<float>(2.2, 10.5),
       legate::Complex<float>(6, 5.98),
       legate::Complex<float>(6, 4),
       legate::Complex<float>(7.9, 12),
       legate::Complex<float>(8, 11)}}}};
  return expect_result;
}

template <typename T, int32_t SIZE, int32_t DIM>
void test_sort(std::array<T, SIZE>& in_array,
               std::array<T, SIZE>& expect,
               legate::Type leg_type,
               std::vector<uint64_t> shape,
               std::optional<int32_t> axis)
{
  auto A1 = cupynumeric::zeros(shape, leg_type);
  if (in_array.size() != 0) {
    if (in_array.size() == 1) {
      A1.fill(legate::Scalar(in_array[0]));
    } else {
      assign_values_to_array<T, DIM>(A1, in_array.data(), in_array.size());
    }
  }
  std::vector<std::string> algos = {"quicksort", "mergesort", "heapsort", "stable"};
  for (auto algo = algos.begin(); algo < algos.end(); ++algo) {
    auto B1 = cupynumeric::sort(A1, axis, *algo);
    if (in_array.size() != 0) {
      check_array_eq<T, DIM>(B1, expect.data(), expect.size());
    }
  }
}

template <typename T>
void sort_basic_axis_impl(std::vector<std::vector<uint64_t>>& test_shapes,
                          std::array<T, 12> in_array,
                          std::vector<std::map<int32_t, std::array<T, 12>>>& expect_result,
                          legate::Type leg_type)
{
  size_t test_shape_size = test_shapes.size();
  for (size_t i = 0; i < test_shape_size; ++i) {
    auto test_shape = test_shapes[i];
    int32_t dim     = test_shape.size();
    for (int32_t axis = -dim + 1; axis < dim; ++axis) {
      auto expect_val = expect_result[i][axis];
      if (dim == 1) {
        test_sort<T, 12, 1>(in_array, expect_val, leg_type, test_shape, axis);
      } else if (dim == 2) {
        test_sort<T, 12, 2>(in_array, expect_val, leg_type, test_shape, axis);
      } else {
        test_sort<T, 12, 3>(in_array, expect_val, leg_type, test_shape, axis);
      }
    }
  }
}

void sort_basic_axis()
{
  std::vector<std::vector<uint64_t>> test_shapes = {
    {12}, {1, 12}, {12, 1}, {3, 4}, {12, 1, 1}, {1, 12, 1}, {1, 1, 12}, {2, 2, 3}};

  // Test int type
  std::array<int32_t, 12> in_array1 = {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1};
  auto expect_result1               = get_expect_result_int();
  sort_basic_axis_impl<int32_t>(test_shapes, in_array1, expect_result1, legate::int32());

  // Test float type
  std::array<double, 12> in_array2 = {1.5, 3.66, 6, 5.98, 2.2, 10.5, 8, 11, 7.9, 12, 9, 4};
  auto expect_result2              = get_expect_result_double();
  sort_basic_axis_impl<double>(test_shapes, in_array2, expect_result2, legate::float64());

  // Test complex type
  std::array<legate::Complex<float>, 12> in_array3 = {legate::Complex<float>(10, 3),
                                                      legate::Complex<float>(12, 5),
                                                      legate::Complex<float>(2, 4),
                                                      legate::Complex<float>(8, 9),
                                                      legate::Complex<float>(7, 6),
                                                      legate::Complex<float>(11, 1),
                                                      legate::Complex<float>(1.5, 3.66),
                                                      legate::Complex<float>(6, 5.98),
                                                      legate::Complex<float>(2.2, 10.5),
                                                      legate::Complex<float>(8, 11),
                                                      legate::Complex<float>(7.9, 12),
                                                      legate::Complex<float>(6, 4)};
  auto expect_result3                              = get_expect_result_complex();
  sort_basic_axis_impl<legate::Complex<float>>(
    test_shapes, in_array3, expect_result3, legate::complex64());
}

void sort_empty_array()
{
  std::vector<std::vector<uint64_t>> test_shapes = {
    {0}, {0, 1}, {1, 0}, {1, 0, 0}, {1, 1, 0}, {1, 0, 1}};

  std::array<int32_t, 0> in_array = {};
  size_t test_shape_size          = test_shapes.size();
  for (size_t i = 0; i < test_shape_size; ++i) {
    auto test_shape = test_shapes[i];
    int32_t dim     = test_shape.size();
    for (int32_t axis = -dim + 1; axis < dim; ++axis) {
      if (dim == 1) {
        test_sort<int32_t, 0, 1>(in_array, in_array, legate::int32(), test_shape, axis);
      } else if (dim == 2) {
        test_sort<int32_t, 0, 2>(in_array, in_array, legate::int32(), test_shape, axis);
      } else {
        test_sort<int32_t, 0, 3>(in_array, in_array, legate::int32(), test_shape, axis);
      }
    }
  }
}

void sort_single_item_array()
{
  std::vector<std::vector<uint64_t>> test_shapes = {{1}, {1, 1}, {1, 1, 1}};

  std::array<int32_t, 1> in_array = {12};
  size_t test_shape_size          = test_shapes.size();
  for (size_t i = 0; i < test_shape_size; ++i) {
    auto test_shape = test_shapes[i];
    int32_t dim     = test_shape.size();
    for (int32_t axis = -dim + 1; axis < dim; ++axis) {
      if (dim == 1) {
        test_sort<int32_t, 1, 1>(in_array, in_array, legate::int32(), test_shape, axis);
      } else if (dim == 2) {
        test_sort<int32_t, 1, 2>(in_array, in_array, legate::int32(), test_shape, axis);
      } else {
        test_sort<int32_t, 1, 3>(in_array, in_array, legate::int32(), test_shape, axis);
      }
    }
  }
}

void sort_negative_test()
{
  auto in_ar1 = cupynumeric::zeros({2, 3}, legate::int32());

  // Test invalid input sort axis
  EXPECT_THROW(cupynumeric::sort(in_ar1, 2, "quicksort"), std::invalid_argument);
  EXPECT_THROW(cupynumeric::sort(in_ar1, -3, "quicksort"), std::invalid_argument);

  // Test invalid input algorithm
  EXPECT_THROW(cupynumeric::sort(in_ar1, 0, "negative"), std::invalid_argument);
}

// void cpp_test()
TEST(Sort, BasicAxis) { sort_basic_axis(); }
TEST(Sort, EmptyArray) { sort_empty_array(); }
TEST(Sort, SingleItemArray) { sort_single_item_array(); }
TEST(Sort, Negative) { sort_negative_test(); }
