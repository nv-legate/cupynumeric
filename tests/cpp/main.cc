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

#include <gtest/gtest.h>
#include "legate.h"
#include "cupynumeric.h"

class Environment : public ::testing::Environment {
 public:
  Environment(int argc, char** argv) : argc_(argc), argv_(argv) {}

  void SetUp() override
  {
    legate::start();
    cupynumeric::initialize(argc_, argv_);
  }
  void TearDown() override { EXPECT_EQ(legate::finish(), 0); }

 private:
  int argc_;
  char** argv_;
};

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new Environment(argc, argv));

  return RUN_ALL_TESTS();
}
