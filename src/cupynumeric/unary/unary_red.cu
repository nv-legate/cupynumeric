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

#include "cupynumeric/unary/unary_red.h"
#include "cupynumeric/unary/unary_red_template.inl"

#include "cupynumeric/cuda_help.h"

namespace cupynumeric {

template <typename T>
static constexpr T div_and_ceil(T value, T divider)
{
  return std::max<T>((value + divider - 1) / divider, 1);
}

static constexpr coord_t WARP_SIZE = 32;

// This helper class is to compute the shape of thread blocks for reduction kernels.
// The strategy is to parallelize on dimensions, from the outermost one to the innermost,
// that are not being collapsed, thereby having threads work on independet lanes of
// reductions as much as possible. In case where the non-collapsing dimensions don't
// have enough elements to be assigned to the threads, we also parallelize on
// the collapsing domain. One exceptional case to this strategy is where the collapsing
// dimension is the innermost one, in which case we prefer that dimension to the others
// in order to enjoy wrap coalescing. The maximum degree of such parallelism would be 32,
// which is the size of a wrap.
template <int32_t DIM>
struct ThreadBlock {
  void initialize(const Rect<DIM>& domain, int32_t collapsed_dim)
  {
    auto remaining = static_cast<coord_t>(THREADS_PER_BLOCK);

    Point<DIM> domain_extents;
    for (int32_t idx = 0; idx < DIM; ++idx) {
      domain_extents[idx] = domain.hi[idx] - domain.lo[idx] + 1;
    }

    // If the innermost dimension is being collapsed, we assign at least one warp to it
    // for warp coalsecing.
    if (collapsed_dim == DIM - 1) {
      auto extent             = std::min<coord_t>(WARP_SIZE, domain_extents[collapsed_dim]);
      extents_[collapsed_dim] = extent;
      remaining               = std::max<coord_t>(remaining / extent, 1);
    }

    // Then, we compute how many threads there should be along aech dimension,
    // excluding the one being collapsed
    for (int32_t idx = DIM - 1; idx >= 0; --idx) {
      if (idx == collapsed_dim) {
        continue;
      }
      auto extent   = std::min(remaining, domain_extents[idx]);
      extents_[idx] = extent;
      remaining     = std::max<coord_t>(remaining / extent, 1);
    }

    // Finally, we determine degree of parallelism for the collapsed dimension if we didn't above
    if (collapsed_dim != DIM - 1) {
      extents_[collapsed_dim] = std::min(remaining, domain_extents[collapsed_dim]);
    }

    // Cache the aggregate number of threads per increment in each dimension,
    // which later will be used for de-linearization of a thread id
    num_threads_ = 1;
    for (int32_t idx = DIM - 1; idx >= 0; --idx) {
      pitches_[idx] = num_threads_;
      num_threads_ *= extents_[idx];
    }
  }

  // Compute a relative coordiate of a given thread
  __host__ __device__ Point<DIM> point(coord_t tid) const
  {
    Point<DIM> p;
    for (int32_t dim = 0; dim < DIM; ++dim) {
      p[dim] = tid / pitches_[dim];
      tid    = tid % pitches_[dim];
    }
    return p;
  }

  // Total number of threads
  size_t num_threads_;
  // Number of threads along each dimension
  Point<DIM> extents_;
  // Aggregate number of threads per increment in each dimension
  Point<DIM> pitches_;
};

// This class represents a set of concurrent thread blocks. Concurrent thread blocks form
// hyperplanes in N-dimensional integer lattice such that the collapsed dimension is normal to them.
// The size of thread blocks is determined by the maximum number of CTAs for a given kernel;
// the number of concurrent thread blocks is the minimum number of hyperplanes whose aggregate
// volume exceeds the maximum number of CTAs.
template <int32_t DIM>
struct ThreadBlocks {
  void initialize(const Rect<DIM>& domain, int32_t collapsed_dim)
  {
    collapsed_dim_ = collapsed_dim;
    block_.initialize(domain, collapsed_dim);

    for (int32_t idx = 0; idx < DIM; ++idx) {
      auto domain_extent = domain.hi[idx] - domain.lo[idx] + 1;
      extents_[idx]      = div_and_ceil(domain_extent, block_.extents_[idx]);
    }

    // We want the collapsed dimension to be the outermost one when
    // de-linearizing the block id.
    dim_order_[0] = collapsed_dim_;
    for (int32_t dim = 0, idx = 1; dim < DIM; ++dim) {
      if (dim != collapsed_dim_) {
        dim_order_[idx++] = dim;
      }
    }

    // Compute the aggregate number of blocks per increment in each dimension
    coord_t num_blocks = 1;
    for (int32_t idx = DIM - 1; idx >= 0; --idx) {
      auto dim      = dim_order_[idx];
      pitches_[dim] = num_blocks;
      num_blocks *= extents_[dim];
    }
    // For now we say all blocks can run concurrent.
    num_blocks_ = num_blocks;
    // Also compute the stride on the collapsed dimension
    collapsed_dim_stride_ = extents_[collapsed_dim_] * block_.extents_[collapsed_dim_];
  }

  // De-linearized the linearized block id and thread it into an N-dimensional point
  __host__ __device__ Point<DIM> point(coord_t bid, coord_t tid, const Point<DIM>& origin) const
  {
    Point<DIM> p = origin;
    for (int32_t dim : dim_order_) {
      p[dim] += (bid / pitches_[dim]) * block_.extents_[dim];
      bid = bid % pitches_[dim];
    }
    p += block_.point(tid);
    return p;
  }

  void compute_maximum_concurrency(const void* func)
  {
    int32_t num_ctas = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_ctas, func, num_threads(), 0);

    size_t plane_size = pitches_[collapsed_dim_];
    // Calculate the number of planes whose volume barely exceeds the maximum number of CTAs
    size_t max_num_concurrent_planes =
      std::max<size_t>(div_and_ceil<size_t>(num_ctas, plane_size), 1);
    // Then we update the number of concurrent thread blocks and the stride on the collapsed
    // dimension
    num_blocks_           = plane_size * max_num_concurrent_planes;
    collapsed_dim_stride_ = max_num_concurrent_planes * block_.extents_[collapsed_dim_];
  }

  __host__ __device__ inline void next_point(Point<DIM>& point) const
  {
    point[collapsed_dim_] += collapsed_dim_stride_;
  }

  constexpr size_t num_blocks() const { return num_blocks_; }
  constexpr size_t num_threads() const { return block_.num_threads_; }

  // List of dimensions, from the outermost one to the innermost
  int32_t dim_order_[DIM];
  int32_t collapsed_dim_;
  coord_t collapsed_dim_stride_;
  // Shape of each thread block
  ThreadBlock<DIM> block_;
  // Number of thread blocks along each dimension
  Point<DIM> extents_;
  // Aggregate number of thread blocks per increment in each dimension
  Point<DIM> pitches_;
  // Number of concurrent thread blocks
  size_t num_blocks_;
};

template <int32_t DIM>
std::ostream& operator<<(std::ostream& os, const ThreadBlock<DIM>& block)
{
  os << "ThreadBlock(extents: " << block.extents_ << ", pitches: " << block.pitches_ << ")";
  return os;
}

template <int32_t DIM>
std::ostream& operator<<(std::ostream& os, const ThreadBlocks<DIM>& blocks)
{
  os << "ThreadBlocks(" << blocks.block_ << ", extents: " << blocks.extents_
     << ", pitches: " << blocks.pitches_ << ", num concurrent blocks: " << blocks.num_blocks_
     << ", dim order: {";
  for (int32_t dim : blocks.dim_order_) {
    os << dim << ", ";
  }
  os << "})";

  return os;
}

template <typename REDOP, typename LHS, int32_t DIM>
static void __device__ __forceinline__ collapse_dims(LHS& result,
                                                     Point<DIM>& point,
                                                     const Rect<DIM>& domain,
                                                     int32_t collapsed_dim,
                                                     LHS identity,
                                                     coord_t tid)
{
#if __CUDA_ARCH__ >= 700
  // If we're collapsing the innermost dimension, we perform some optimization
  // with shared memory to reduce memory traffic due to atomic updates
  if (collapsed_dim == DIM - 1) {
    __shared__ uint8_t shmem[THREADS_PER_BLOCK * sizeof(LHS)];
    LHS* trampoline = reinterpret_cast<LHS*>(shmem);
    // Check for the case where all the threads in the same warp have
    // the same x value in which case they're all going to conflict
    // so instead we do a warp-level reduction so just one thread ends
    // up doing the full atomic
    coord_t bucket = 0;
    for (int32_t dim = DIM - 2; dim >= 0; --dim) {
      bucket = bucket * (domain.hi[dim] - domain.lo[dim] + 1) + point[dim] - domain.lo[dim];
    }

    const uint32_t same_mask = __match_any_sync(0xffffffff, bucket);
    int32_t laneid;
    asm volatile("mov.s32 %0, %laneid;" : "=r"(laneid));
    const uint32_t active_mask = __ballot_sync(0xffffffff, same_mask - (1 << laneid));
    if ((active_mask & (1 << laneid)) != 0) {
      // Store our data into shared
      trampoline[tid] = result;
      // Make sure all the threads in the warp are done writing
      __syncwarp(active_mask);
      // Have the lowest thread in each mask pull in the values
      int32_t lowest_index = -1;
      for (int32_t i = 0; i < warpSize; i++) {
        if (same_mask & (1 << i)) {
          if (lowest_index == -1) {
            if (i != laneid) {
              // We're not the lowest thread in the warp for
              // this value so we're done, set the value back
              // to identity to ensure that we don't try to
              // perform the reduction out to memory
              result = identity;
              break;
            } else {  // Make sure we don't do this test again
              lowest_index = i;
            }
            // It was already our value, so just keep going
          } else {
            // Pull in the value from shared memory
            const int32_t index = tid + i - laneid;
            REDOP::template fold<true>(result, trampoline[index]);
          }
        }
      }
    }
  }
#endif

  if (LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)) {
    // Note: this isn't necessary because we know that the affine transformation on the output
    // accessor will ignore coordinates of the collapsed dimension. However, Legion's bounds checks
    // want the accessor to honor the sub-rectangle passed when it was created, so we need to
    // put points back in the bounds to appease the checks.
    point[collapsed_dim] = domain.lo[collapsed_dim];
  }
}

template <typename OP, typename REDOP, typename LHS, typename RHS, int32_t DIM, bool HAS_WHERE>
static __device__ __forceinline__ Point<DIM> local_reduce(LHS& result,
                                                          AccessorRO<RHS, DIM> in,
                                                          AccessorRO<bool, DIM> where,
                                                          LHS identity,
                                                          const ThreadBlocks<DIM>& blocks,
                                                          const Rect<DIM>& domain,
                                                          int32_t collapsed_dim)
{
  const coord_t tid = threadIdx.x;
  const coord_t bid = blockIdx.x;

  Point<DIM> point = blocks.point(bid, tid, domain.lo);
  if (!domain.contains(point)) {
    return point;
  }

  bool mask = true;
  while (point[collapsed_dim] <= domain.hi[collapsed_dim]) {
    if constexpr (HAS_WHERE) {
      mask = where[point];
    }
    if (mask) {
      LHS value = OP::convert(point, collapsed_dim, identity, in[point]);
      REDOP::template fold<true>(result, value);
    }
    blocks.next_point(point);
  }

  collapse_dims<REDOP, LHS>(result, point, domain, collapsed_dim, identity, tid);
  return point;
}

template <typename OP, typename REDOP, typename LHS, typename RHS, int32_t DIM, bool HAS_WHERE>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  reduce_with_rd_acc(AccessorRD<REDOP, false, DIM> out,
                     AccessorRO<RHS, DIM> in,
                     AccessorRO<bool, DIM> where,
                     LHS identity,
                     ThreadBlocks<DIM> blocks,
                     Rect<DIM> domain,
                     int32_t collapsed_dim)
{
  auto result = identity;
  auto point  = local_reduce<OP, REDOP, LHS, RHS, DIM, HAS_WHERE>(
    result, in, where, identity, blocks, domain, collapsed_dim);
  if (result != identity) {
    out.reduce(point, result);
  }
}

template <UnaryRedCode OP_CODE, Type::Code CODE, int DIM, bool HAS_WHERE>
struct UnaryRedImplBody<VariantKind::GPU, OP_CODE, CODE, DIM, HAS_WHERE> {
  TaskContext context;
  explicit UnaryRedImplBody(TaskContext context) : context(context) {}

  using OP    = UnaryRedOp<OP_CODE, CODE>;
  using LG_OP = typename OP::OP;
  using RHS   = type_of<CODE>;
  using LHS   = typename OP::VAL;

  void operator()(AccessorRD<LG_OP, false, DIM> lhs,
                  AccessorRO<RHS, DIM> rhs,
                  AccessorRO<bool, DIM> where,
                  const Rect<DIM>& rect,
                  const Pitches<DIM - 1>& pitches,
                  int collapsed_dim,
                  size_t volume) const
  {
    auto Kernel = reduce_with_rd_acc<OP, LG_OP, LHS, RHS, DIM, HAS_WHERE>;
    auto stream = context.get_task_stream();

    ThreadBlocks<DIM> blocks;
    blocks.initialize(rect, collapsed_dim);

    blocks.compute_maximum_concurrency(reinterpret_cast<const void*>(Kernel));
    Kernel<<<blocks.num_blocks(), blocks.num_threads(), 0, stream>>>(
      lhs, rhs, where, LG_OP::identity, blocks, rect, collapsed_dim);
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void UnaryRedTask::gpu_variant(TaskContext context)
{
  unary_red_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
