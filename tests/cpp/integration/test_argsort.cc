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

auto get_argsort_expect_result()
{
  std::vector<std::map<int32_t, std::array<int64_t, 12>>> expect_result = {
    {{0, {11, 4, 1, 5, 3, 9, 8, 6, 7, 0, 10, 2}}},
    {{-1, {11, 4, 1, 5, 3, 9, 8, 6, 7, 0, 10, 2}},
     {0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {1, {11, 4, 1, 5, 3, 9, 8, 6, 7, 0, 10, 2}}},
    {{-1, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {0, {11, 4, 1, 5, 3, 9, 8, 6, 7, 0, 10, 2}},
     {1, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}},
    {{-1, {1, 3, 0, 2, 0, 1, 2, 3, 3, 1, 0, 2}},
     {0, {1, 0, 1, 2, 2, 1, 2, 0, 0, 2, 0, 1}},
     {1, {1, 3, 0, 2, 0, 1, 2, 3, 3, 1, 0, 2}}},
    {{-2, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {-1, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {0, {11, 4, 1, 5, 3, 9, 8, 6, 7, 0, 10, 2}},
     {1, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {2, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}},
    {{-2, {11, 4, 1, 5, 3, 9, 8, 6, 7, 0, 10, 2}},
     {-1, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {1, {11, 4, 1, 5, 3, 9, 8, 6, 7, 0, 10, 2}},
     {2, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}},
    {{-2, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {-1, {11, 4, 1, 5, 3, 9, 8, 6, 7, 0, 10, 2}},
     {0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {1, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {2, {11, 4, 1, 5, 3, 9, 8, 6, 7, 0, 10, 2}}},
    {{-2, {1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0}},
     {-1, {1, 0, 2, 1, 2, 0, 2, 0, 1, 2, 0, 1}},
     {0, {1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0}},
     {1, {1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0}},
     {2, {1, 0, 2, 1, 2, 0, 2, 0, 1, 2, 0, 1}}}};
  return expect_result;
}

auto get_argsort_expect_result_4d()
{
  std::vector<std::map<int32_t, std::array<int64_t, 16>>> expect_result = {
    {{-3, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {-2, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {-1, {14, 6, 2, 7, 4, 12, 11, 9, 10, 1, 13, 3, 5, 0, 15, 8}},
     {0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {1, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {2, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {3, {14, 6, 2, 7, 4, 12, 11, 9, 10, 1, 13, 3, 5, 0, 15, 8}}},
    {{-3, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {-2, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {-1, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {0, {14, 6, 2, 7, 4, 12, 11, 9, 10, 1, 13, 3, 5, 0, 15, 8}},
     {1, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {2, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {3, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}},
    {{-3, {1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1}},
     {-2, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {-1, {2, 1, 3, 0, 2, 3, 0, 1, 3, 1, 2, 0, 2, 0, 1, 3}},
     {0, {0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1}},
     {1, {1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1}},
     {2, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {3, {2, 1, 3, 0, 2, 3, 0, 1, 3, 1, 2, 0, 2, 0, 1, 3}}}};
  return expect_result;
}

auto get_argsort_expect_result_5d()
{
  std::vector<std::map<int32_t, std::array<int64_t, 16>>> expect_result = {
    {{-4, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {-3, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {-2, {14, 6, 2, 7, 4, 12, 11, 9, 10, 1, 13, 3, 5, 0, 15, 8}},
     {-1, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {1, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {2, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {3, {14, 6, 2, 7, 4, 12, 11, 9, 10, 1, 13, 3, 5, 0, 15, 8}},
     {4, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}},
    {{-4, {14, 6, 2, 7, 4, 12, 11, 9, 10, 1, 13, 3, 5, 0, 15, 8}},
     {-3, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {-2, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {-1, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {1, {14, 6, 2, 7, 4, 12, 11, 9, 10, 1, 13, 3, 5, 0, 15, 8}},
     {2, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {3, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {4, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}},
    {{-4, {0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1}},
     {-3, {1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1}},
     {-2, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {-1, {2, 1, 3, 0, 2, 3, 0, 1, 3, 1, 2, 0, 2, 0, 1, 3}},
     {0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {1, {0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1}},
     {2, {1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1}},
     {3, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {4, {2, 1, 3, 0, 2, 3, 0, 1, 3, 1, 2, 0, 2, 0, 1, 3}}}};
  return expect_result;
}

auto get_argsort_expect_result_6d()
{
  std::vector<std::map<int32_t, std::array<int64_t, 16>>> expect_result = {
    {{-5, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {-4, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {-3, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {-2, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {-1, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {0, {14, 6, 2, 7, 4, 12, 11, 9, 10, 1, 13, 3, 5, 0, 15, 8}},
     {1, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {2, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {3, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {4, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {5, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}},
    {{-5, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {-4, {14, 6, 2, 7, 4, 12, 11, 9, 10, 1, 13, 3, 5, 0, 15, 8}},
     {-3, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {-2, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {-1, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {1, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {2, {14, 6, 2, 7, 4, 12, 11, 9, 10, 1, 13, 3, 5, 0, 15, 8}},
     {3, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {4, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {5, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}},
    {{-5, {0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1}},
     {-4, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {-3, {1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1}},
     {-2, {1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1}},
     {-1, {1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1}},
     {0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {1, {0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1}},
     {2, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {3, {1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1}},
     {4, {1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1}},
     {5, {1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1}}}};
  return expect_result;
}

auto get_argsort_expect_result_7d()
{
  std::vector<std::map<int32_t, std::array<int64_t, 16>>> expect_result = {
    {{-6, {14, 6, 2, 7, 4, 12, 11, 9, 10, 1, 13, 3, 5, 0, 15, 8}},
     {-5, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {-4, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {-3, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {-2, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {-1, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {1, {14, 6, 2, 7, 4, 12, 11, 9, 10, 1, 13, 3, 5, 0, 15, 8}},
     {2, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {3, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {4, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {5, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {6, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}},
    {{-6, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {-5, {0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1}},
     {-4, {1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1}},
     {-3, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {-2, {2, 1, 3, 0, 2, 3, 0, 1, 3, 1, 2, 0, 2, 0, 1, 3}},
     {-1, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {1, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {2, {0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1}},
     {3, {1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1}},
     {4, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {5, {2, 1, 3, 0, 2, 3, 0, 1, 3, 1, 2, 0, 2, 0, 1, 3}},
     {6, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}},
    {{-6, {1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1}},
     {-5, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {-4, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {-3, {1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1}},
     {-2, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {-1, {1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1}},
     {0, {0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1}},
     {1, {1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1}},
     {2, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {3, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {4, {1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1}},
     {5, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
     {6, {1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1}}}};
  return expect_result;
}

template <typename T, int32_t SIZE, int32_t DIM>
void test_argsort(std::array<T, SIZE>& in_array,
                  std::array<int64_t, SIZE>& expect,
                  legate::Type leg_type,
                  std::vector<uint64_t> shape,
                  std::optional<int32_t> axis,
                  bool test_only_stable = false)
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
  if (test_only_stable) {
    algos = {"mergesort", "stable"};
  }
  for (auto algo = algos.begin(); algo < algos.end(); ++algo) {
    auto B1 = cupynumeric::argsort(A1, axis, *algo);
    if (in_array.size() != 0) {
      check_array_eq<int64_t, DIM>(B1, expect.data(), expect.size());
    }
  }
}

template <typename T, int32_t SIZE>
void argsort_basic_axis_impl(
  std::vector<std::vector<uint64_t>>& test_shapes,
  std::array<T, SIZE> in_array,
  std::vector<std::map<int32_t, std::array<int64_t, SIZE>>>& expect_result,
  legate::Type leg_type,
  bool test_only_stable = false)
{
  size_t test_shape_size = test_shapes.size();
  for (size_t i = 0; i < test_shape_size; ++i) {
    auto test_shape = test_shapes[i];
    int32_t dim     = test_shape.size();
    for (int32_t axis = -dim + 1; axis < dim; ++axis) {
      auto expect_val = expect_result[i][axis];
      if (dim == 1) {
        test_argsort<T, SIZE, 1>(
          in_array, expect_val, leg_type, test_shape, axis, test_only_stable);
      } else if (dim == 2) {
        test_argsort<T, SIZE, 2>(
          in_array, expect_val, leg_type, test_shape, axis, test_only_stable);
      } else if (dim == 3) {
        test_argsort<T, SIZE, 3>(
          in_array, expect_val, leg_type, test_shape, axis, test_only_stable);
      } else if (dim == 4) {
#if LEGATE_MAX_DIM >= 4
        test_argsort<T, SIZE, 4>(
          in_array, expect_val, leg_type, test_shape, axis, test_only_stable);
#endif
      } else if (dim == 5) {
#if LEGATE_MAX_DIM >= 5
        test_argsort<T, SIZE, 5>(
          in_array, expect_val, leg_type, test_shape, axis, test_only_stable);
#endif
      } else if (dim == 6) {
#if LEGATE_MAX_DIM >= 6
        test_argsort<T, SIZE, 6>(
          in_array, expect_val, leg_type, test_shape, axis, test_only_stable);
#endif
      } else if (dim == 7) {
#if LEGATE_MAX_DIM >= 7
        test_argsort<T, SIZE, 7>(
          in_array, expect_val, leg_type, test_shape, axis, test_only_stable);
#endif
      }
    }
  }
}

void argsort_basic_axis()
{
  std::vector<std::vector<uint64_t>> test_shapes = {
    {12}, {1, 12}, {12, 1}, {3, 4}, {12, 1, 1}, {1, 12, 1}, {1, 1, 12}, {2, 2, 3}};

  auto expect_result = get_argsort_expect_result();

  // Test int type
  std::array<int32_t, 12> in_array1 = {10, 3, 12, 5, 2, 4, 8, 9, 7, 6, 11, 1};
  argsort_basic_axis_impl<int32_t, 12>(test_shapes, in_array1, expect_result, legate::int32());

  // Test float type
  std::array<double, 12> in_array2 = {10.5, 3.66, 12, 5.98, 2.2, 4, 8, 9, 7.9, 6, 11, 1.5};
  argsort_basic_axis_impl<double, 12>(test_shapes, in_array2, expect_result, legate::float64());

  // Test complex type
  std::array<legate::Complex<float>, 12> in_array3 = {legate::Complex<float>(10, 3),
                                                      legate::Complex<float>(2.2, 10.5),
                                                      legate::Complex<float>(12, 5),
                                                      legate::Complex<float>(6, 5.98),
                                                      legate::Complex<float>(2, 4),
                                                      legate::Complex<float>(6, 4),
                                                      legate::Complex<float>(8, 9),
                                                      legate::Complex<float>(8, 11),
                                                      legate::Complex<float>(7.9, 12),
                                                      legate::Complex<float>(7, 6),
                                                      legate::Complex<float>(11, 1),
                                                      legate::Complex<float>(1.5, 3.66)};
  argsort_basic_axis_impl<legate::Complex<float>, 12>(
    test_shapes, in_array3, expect_result, legate::complex64());
}

void argsort_basic_axis_stable()
{
  std::vector<std::vector<uint64_t>> test_shapes = {
    {12}, {1, 12}, {12, 1}, {3, 4}, {12, 1, 1}, {1, 12, 1}, {1, 1, 12}, {2, 2, 3}};
  auto expect_result = get_argsort_expect_result();

  // Test int type
  std::array<int32_t, 12> in_array1 = {10, 3, 12, 5, 2, 3, 8, 8, 7, 6, 10, 1};
  argsort_basic_axis_impl<int32_t, 12>(
    test_shapes, in_array1, expect_result, legate::int32(), true);

  // Test float type
  std::array<double, 12> in_array2 = {10.5, 3.66, 12, 5.98, 2.2, 3.66, 8, 9, 7.9, 6, 10.5, 1.5};
  argsort_basic_axis_impl<double, 12>(
    test_shapes, in_array2, expect_result, legate::float64(), true);

  // Test complex type
  std::array<legate::Complex<float>, 12> in_array3 = {legate::Complex<float>(10, 3),
                                                      legate::Complex<float>(2.2, 10.5),
                                                      legate::Complex<float>(12, 5),
                                                      legate::Complex<float>(6, 5.98),
                                                      legate::Complex<float>(2, 4),
                                                      legate::Complex<float>(2.2, 10.5),
                                                      legate::Complex<float>(8, 9),
                                                      legate::Complex<float>(8, 11),
                                                      legate::Complex<float>(7.9, 12),
                                                      legate::Complex<float>(7, 6),
                                                      legate::Complex<float>(10, 3),
                                                      legate::Complex<float>(1.5, 3.66)};
  argsort_basic_axis_impl<legate::Complex<float>, 12>(
    test_shapes, in_array3, expect_result, legate::complex64(), true);
}

void argsort_basic_axis_max_dim()
{
  // Only test int type for max dim
  std::array<int32_t, 16> in_array = {14, 10, 3, 12, 5, 13, 2, 4, 16, 8, 9, 7, 6, 11, 1, 15};
#if LEGATE_MAX_DIM >= 4
  std::vector<std::vector<uint64_t>> test_shapes_4d = {{1, 1, 1, 16}, {16, 1, 1, 1}, {2, 2, 1, 4}};
  auto expect_result_4d                             = get_argsort_expect_result_4d();
  argsort_basic_axis_impl<int32_t, 16>(test_shapes_4d, in_array, expect_result_4d, legate::int32());
#endif

#if LEGATE_MAX_DIM >= 5
  std::vector<std::vector<uint64_t>> test_shapes_5d = {
    {1, 1, 1, 16, 1}, {1, 16, 1, 1, 1}, {1, 2, 2, 1, 4}};
  auto expect_result_5d = get_argsort_expect_result_5d();
  argsort_basic_axis_impl<int32_t, 16>(test_shapes_5d, in_array, expect_result_5d, legate::int32());
#endif

#if LEGATE_MAX_DIM >= 6
  std::vector<std::vector<uint64_t>> test_shapes_6d = {
    {16, 1, 1, 1, 1, 1}, {1, 1, 16, 1, 1, 1}, {1, 2, 1, 2, 2, 2}};
  auto expect_result_6d = get_argsort_expect_result_6d();
  argsort_basic_axis_impl<int32_t, 16>(test_shapes_6d, in_array, expect_result_6d, legate::int32());
#endif

#if LEGATE_MAX_DIM >= 7
  std::vector<std::vector<uint64_t>> test_shapes_7d = {
    {1, 16, 1, 1, 1, 1, 1}, {1, 1, 2, 2, 1, 4, 1}, {2, 2, 1, 1, 2, 1, 2}};
  auto expect_result_7d = get_argsort_expect_result_7d();
  argsort_basic_axis_impl<int32_t, 16>(test_shapes_7d, in_array, expect_result_7d, legate::int32());
#endif
}

void argsort_large_array()
{
  const int32_t count                            = 10000;
  std::vector<std::vector<uint64_t>> test_shapes = {{count}};
  std::array<int64_t, count> expect_val;
  for (int64_t j = 0; j < count; j++) {
    expect_val[j] = count - 1 - j;
  }
  std::vector<std::map<int32_t, std::array<int64_t, count>>> expect_result = {{{0, expect_val}}};

  // Test int type for large array
  std::array<int32_t, count> in_array1;
  for (int32_t i = 0; i < count; i++) {
    in_array1[i] = count - i;
  }
  argsort_basic_axis_impl<int32_t, count>(test_shapes, in_array1, expect_result, legate::int32());

  // Test float type
  std::array<double, count> in_array2;
  for (int32_t i = 0; i < count; i++) {
    in_array2[i] = count * 1.1 - i;
  }
  argsort_basic_axis_impl<double, count>(test_shapes, in_array2, expect_result, legate::float64());

  // Test complex type
  std::array<legate::Complex<float>, count> in_array3;
  for (int32_t i = 0; i < count; i++) {
    in_array3[i] = legate::Complex<float>(count - i, count - i);
  }
  argsort_basic_axis_impl<legate::Complex<float>, count>(
    test_shapes, in_array3, expect_result, legate::complex64());
}

void argsort_empty_array()
{
  std::vector<std::vector<uint64_t>> test_shapes = {
    {0}, {0, 1}, {1, 0}, {1, 0, 0}, {1, 1, 0}, {1, 0, 1}};

  std::array<int32_t, 0> in_array   = {};
  std::array<int64_t, 0> expect_val = {};
  size_t test_shape_size            = test_shapes.size();
  for (size_t i = 0; i < test_shape_size; ++i) {
    auto test_shape = test_shapes[i];
    int32_t dim     = test_shape.size();
    for (int32_t axis = -dim + 1; axis < dim; ++axis) {
      if (dim == 1) {
        test_argsort<int32_t, 0, 1>(in_array, expect_val, legate::int32(), test_shape, axis);
      } else if (dim == 2) {
        test_argsort<int32_t, 0, 2>(in_array, expect_val, legate::int32(), test_shape, axis);
      } else {
        test_argsort<int32_t, 0, 3>(in_array, expect_val, legate::int32(), test_shape, axis);
      }
    }
  }
}

void argsort_single_item_array()
{
  std::vector<std::vector<uint64_t>> test_shapes = {{1}, {1, 1}, {1, 1, 1}};

  std::array<int32_t, 1> in_array   = {12};
  std::array<int64_t, 1> expect_val = {0};
  size_t test_shape_size            = test_shapes.size();
  for (size_t i = 0; i < test_shape_size; ++i) {
    auto test_shape = test_shapes[i];
    int32_t dim     = test_shape.size();
    for (int32_t axis = -dim + 1; axis < dim; ++axis) {
      if (dim == 1) {
        test_argsort<int32_t, 1, 1>(in_array, expect_val, legate::int32(), test_shape, axis);
      } else if (dim == 2) {
        test_argsort<int32_t, 1, 2>(in_array, expect_val, legate::int32(), test_shape, axis);
      } else {
        test_argsort<int32_t, 1, 3>(in_array, expect_val, legate::int32(), test_shape, axis);
      }
    }
  }
}

void argsort_negative_test()
{
  auto in_ar1 = cupynumeric::zeros({2, 3}, legate::int32());

  // Test invalid input sort axis
  EXPECT_THROW(cupynumeric::argsort(in_ar1, 2, "quicksort"), std::invalid_argument);
  EXPECT_THROW(cupynumeric::argsort(in_ar1, -3, "quicksort"), std::invalid_argument);

  // Test invalid input algorithm
  EXPECT_THROW(cupynumeric::argsort(in_ar1, 0, "negative"), std::invalid_argument);
}

// void cpp_test()
TEST(Argsort, BasicAxis) { argsort_basic_axis(); }
TEST(Argsort, BasicAxisStable) { argsort_basic_axis_stable(); }
TEST(Argsort, BasicAxisMaxDim) { argsort_basic_axis_max_dim(); }
TEST(Argsort, LargeArray) { argsort_large_array(); }
TEST(Argsort, EmptyArray) { argsort_empty_array(); }
TEST(Argsort, SingleItemArray) { argsort_single_item_array(); }
TEST(Argsort, Negative) { argsort_negative_test(); }
