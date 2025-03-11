/* Copyright 2023 NVIDIA Corporation
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

#include <gtest/gtest.h>
#include "common_utils.h"

using namespace cupynumeric;

namespace {

struct windows_case {
  int64_t input;
  std::vector<double> expected_values;
  std::vector<uint64_t> expected_shape;
};

struct kaiser_case {
  int64_t input;
  double beta_input;
  std::vector<double> expected_values;
  std::vector<uint64_t> expected_shape;
};

class NormalInput : public ::testing::Test, public ::testing::WithParamInterface<windows_case> {};
class BartlettTest : public NormalInput {};
class BlackmanTest : public NormalInput {};
class HammingTest : public NormalInput {};
class HanningTest : public NormalInput {};
class KaiserTest : public ::testing::Test, public ::testing::WithParamInterface<kaiser_case> {};

INSTANTIATE_TEST_SUITE_P(WindowsTests,
                         BartlettTest,
                         ::testing::Values(windows_case{-1, {}, {0}},
                                           windows_case{0, {}, {0}},
                                           windows_case{1, {1}, {1}},
                                           windows_case{10,
                                                        {0.0,
                                                         0.22222222,
                                                         0.44444444,
                                                         0.66666667,
                                                         0.88888889,
                                                         0.88888889,
                                                         0.66666667,
                                                         0.44444444,
                                                         0.22222222,
                                                         0.0},
                                                        {10}}));

INSTANTIATE_TEST_SUITE_P(WindowsTests,
                         BlackmanTest,
                         ::testing::Values(windows_case{-1, {}, {0}},
                                           windows_case{0, {}, {0}},
                                           windows_case{1, {1}, {1}},
                                           windows_case{10,
                                                        {-1.38777878e-17,
                                                         5.08696327e-02,
                                                         2.58000502e-01,
                                                         6.30000000e-01,
                                                         9.51129866e-01,
                                                         9.51129866e-01,
                                                         6.30000000e-01,
                                                         2.58000502e-01,
                                                         5.08696327e-02,
                                                         -1.38777878e-17},
                                                        {10}}));

INSTANTIATE_TEST_SUITE_P(WindowsTests,
                         HammingTest,
                         ::testing::Values(windows_case{-1, {}, {0}},
                                           windows_case{0, {}, {0}},
                                           windows_case{1, {1}, {1}},
                                           windows_case{10,
                                                        {0.08,
                                                         0.18761956,
                                                         0.46012184,
                                                         0.77,
                                                         0.97225861,
                                                         0.97225861,
                                                         0.77,
                                                         0.46012184,
                                                         0.18761956,
                                                         0.08},
                                                        {10}}));

INSTANTIATE_TEST_SUITE_P(WindowsTests,
                         HanningTest,
                         ::testing::Values(windows_case{-1, {}, {0}},
                                           windows_case{0, {}, {0}},
                                           windows_case{1, {1}, {1}},
                                           windows_case{10,
                                                        {0.0,
                                                         0.11697778,
                                                         0.41317591,
                                                         0.75,
                                                         0.96984631,
                                                         0.96984631,
                                                         0.75,
                                                         0.41317591,
                                                         0.11697778,
                                                         0.0},
                                                        {10}}));

INSTANTIATE_TEST_SUITE_P(WindowsTests,
                         KaiserTest,
                         ::testing::Values(kaiser_case{-1, -1.0, {}, {0}},
                                           kaiser_case{-1, 0, {}, {0}},
                                           kaiser_case{-1, 5, {}, {0}},
                                           kaiser_case{0, -1.0, {}, {0}},
                                           kaiser_case{0, 0, {}, {0}},
                                           kaiser_case{0, 5, {}, {0}},
                                           kaiser_case{1, -1.0, {1}, {1}},
                                           kaiser_case{1, 0, {1}, {1}},
                                           kaiser_case{1, 5, {1}, {1}},
                                           kaiser_case{10,
                                                       -1.0,
                                                       {0.78984831,
                                                        0.86980546,
                                                        0.93237871,
                                                        0.97536552,
                                                        0.99724655,
                                                        0.99724655,
                                                        0.97536552,
                                                        0.93237871,
                                                        0.86980546,
                                                        0.78984831},
                                                       {10}},
                                           kaiser_case{
                                             10, 0, {1., 1., 1., 1., 1., 1., 1., 1., 1., 1.}, {10}},
                                           kaiser_case{10,
                                                       5,
                                                       {0.03671089,
                                                        0.20127873,
                                                        0.47552746,
                                                        0.7753221,
                                                        0.97273069,
                                                        0.97273069,
                                                        0.7753221,
                                                        0.47552746,
                                                        0.20127873,
                                                        0.03671089},
                                                       {10}}));

TEST_P(BartlettTest, Basic)
{
  auto& [input, expected_values, expected_shape] = GetParam();

  auto result = cupynumeric::bartlett(input);
  check_array_near(result, expected_values, expected_shape);
}

TEST_P(BlackmanTest, Basic)
{
  auto& [input, expected_values, expected_shape] = GetParam();

  auto result = cupynumeric::blackman(input);
  check_array_near(result, expected_values, expected_shape);
}

TEST_P(HammingTest, Basic)
{
  auto& [input, expected_values, expected_shape] = GetParam();

  auto result = cupynumeric::hamming(input);
  check_array_near(result, expected_values, expected_shape);
}

TEST_P(HanningTest, Basic)
{
  auto& [input, expected_values, expected_shape] = GetParam();

  auto result = cupynumeric::hanning(input);
  check_array_near(result, expected_values, expected_shape);
}

TEST_P(KaiserTest, Basic)
{
  auto& [input, beta_input, expected_values, expected_shape] = GetParam();

  auto result = cupynumeric::kaiser(input, beta_input);
  check_array_near(result, expected_values, expected_shape);
}

}  // namespace
