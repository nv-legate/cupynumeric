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

#include "generator.cuh"

#include "generator_exponential.inl"
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, exponential_t<float>, float>;
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, exponential_t<double>, double>;

#include "generator_gumbel.inl"
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, gumbel_t<float>, float>;
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, gumbel_t<double>, double>;

#include "generator_laplace.inl"
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, laplace_t<float>, float>;
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, laplace_t<double>, double>;

#include "generator_logistic.inl"
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, logistic_t<float>, float>;
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, logistic_t<double>, double>;

#include "generator_pareto.inl"
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, pareto_t<float>, float>;
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, pareto_t<double>, double>;

#include "generator_power.inl"
template struct randutilimpl::dispatcher<randutilimpl::execlocation::DEVICE, power_t<float>, float>;
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, power_t<double>, double>;

#include "generator_rayleigh.inl"
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, rayleigh_t<float>, float>;
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, rayleigh_t<double>, double>;

#include "generator_cauchy.inl"
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, cauchy_t<float>, float>;
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, cauchy_t<double>, double>;

#include "generator_triangular.inl"
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, triangular_t<float>, float>;
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, triangular_t<double>, double>;

#include "generator_weibull.inl"
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, weibull_t<float>, float>;
template struct randutilimpl::
  dispatcher<randutilimpl::execlocation::DEVICE, weibull_t<double>, double>;
