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

// These values are copied manually in cupynumeric.settings and there is a Python
// unit test that will maintain that these values and the Python settings
// values agree. If these values are modified, the corresponding Python values
// must also be updated.

// 1 << 13 (need actual number for python to parse)
#define MAX_EAGER_VOLUME_DEFAULT 8192
#define MAX_EAGER_VOLUME_TEST 2

// 1 << 27 (need actual number for python to parse)
#define MATMUL_CACHE_SIZE_DEFAULT 134217728
#define MATMUL_CACHE_SIZE_TEST 4096
