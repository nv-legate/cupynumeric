.. _advanced_profiling_debugging:

Advanced Topics in cuPyNumeric 
(Profiling & Debugging)
======================================================

Overview
--------

This section assumes familiarity with running cuPyNumeric, extending it with
Legate Task, and scaling gradient boosting with Legate Boost. For a refresher,
see:

- Setting up your environment and running cuPyNumeric
- Extending cuPyNumeric with Legate Task
- Scaling gradient boosting with Legate Boost

cuPyNumeric scales familiar NumPy workloads seamlessly across CPUs, GPUs, and
multi-node clusters. Previous sections covered how to get code running; here the
focus shifts to making workloads production-ready. At scale, success is not just
about adding GPUs or nodes, it requires ensuring that applications remain
efficient, stable, and resilient under load. That means finding bottlenecks,
managing memory effectively, and preventing failures before they disrupt a job.

This section focuses on two advanced capabilities in cuPyNumeric and the Legate
runtime that address these challenges:

- **Profiling cuPyNumeric applications** – tune performance and analyze
  scalability. Profiling reveals bottlenecks such as idle GPUs,
  synchronization delays, or overly fine-grained tasks, helping you restructure
  code for better scaling.
- **Debugging and Out-of-Memory (OOM) strategies** – improve reliability in
  memory-intensive workloads. These tools help diagnose crashes, manage GPU/CPU
  memory effectively, and prevent common anti-patterns so applications remain
  robust under heavy loads.

By combining profiling tools with solid OOM-handling strategies, you can
significantly improve the efficiency, scalability, and reliability of
cuPyNumeric applications across CPUs, GPUs, and multi-node systems.

For more detail, see the official references:

- cuPyNumeric Documentation
- Legate Documentation


Using the Legate Profiler
-------------------------

Installation
~~~~~~~~~~~~

1. To install the built-in Legate profiler tool in your Conda environment, run:

.. code-block:: bash

   conda install -c conda-forge -c legate legate-profiler

Profiling runs
~~~~~~~~~~~~~~

2. After installing the Legate profiler (``legate-profiler``), profile the code
   using the ``--profile`` flag:

.. code-block:: bash

   # CPU example
   legate --cpus 8 --sysmem 4000 --profile myprog.py

   # Single GPU example
   legate --gpus 1 --profile myprog.py

   # Multi-GPU example (single node, multi-rank: 4 ranks × 1 GPU)
   srun -n 4 --mpi=pmix \
        legate --gpus 1 --profile myprog.py

   # Multi-node example (2 nodes × 4 GPUs = 8 ranks × 1 GPU)
   srun -N 2 --ntasks-per-node=4 \
        --gpus-per-task=1 --gpu-bind=single:1 \
        --mpi=pmix -C gpu \
     legate --gpus 1 --profile myprog.py

3. Similarly, a program can be run via the ``LEGATE_CONFIG`` environment
   variable:

.. code-block:: bash

   LEGATE_CONFIG="--cpus 8 --sysmem 4000 --profile" python ./myprog.py

Profiler outputs
~~~~~~~~~~~~~~~~

4. After a run completes, in the directory you ran the command you’ll see:

- A folder: ``legate_prof/`` – a self-contained HTML report
  (open ``legate_prof/index.html``)
- One or more raw trace files: ``legate_*.prof`` (one per rank)

The ``legate_*.prof`` files are what you need to view locally on your machine.

Examples:

.. code-block:: text

   # CPU / Single GPU - 1 rank
   legate_0.prof
   legate_prof/

   # Multi-GPU - 4 ranks
   legate_0.prof
   legate_1.prof
   legate_2.prof
   legate_3.prof
   legate_prof/

   # Multi-Node (e.g., 2 nodes × 4 GPUs = 8 ranks)
   legate_0.prof ... legate_7.prof
   legate_prof/

.. note::

   Trace files are numbered by rank index (e.g., ``legate_0.prof``,
   ``legate_1.prof``), not by run. If you run again in the same directory,
   files with the same rank numbers will be overwritten; for example, a
   1-rank run will replace ``legate_0.prof``. The ``legate_prof/`` HTML
   report directory is also overwritten.

Local setup: WSL, Miniforge, and profile viewer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

5. On Linux or WSL2, you can set up a local environment to view profiles:

.. code-block:: bash

   # Download Miniforge installer
   wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

   # Install into home
   bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"

   # Load Conda into current shell
   source "$HOME/miniforge3/etc/profile.d/conda.sh"

   # Configure future shells
   conda init bash

   # Restart shell to apply changes
   exec $SHELL -l

   # Create and activate an environment with Legate + cuPyNumeric + profiler
   conda create -n legate -y -c conda-forge -c legate legate cupynumeric legate-profiler
   conda activate legate

Copying profiler files from a remote cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

6. Create a single top-level folder on your local machine and keep runs
   separated to avoid name/file clashes:

.. code-block:: bash

   # Copy legate_*.prof file(s) for CPU, single-GPU, multi-GPU, or multi-node
   scp -r <USER>@<REMOTE_HOST>:<REMOTE_RUN_DIR>/legate_*.prof \
         "<LOCAL_DIR>/<FOLDER_NAME>/name_of_run"

Viewing profiles locally
~~~~~~~~~~~~~~~~~~~~~~~~

7. On your local machine, use:

.. code-block:: bash

   # CPU/GPU: single file (rank 0)
   legate_prof view /path/to/legate_0.prof

   # Multi-GPU/Multi-Node: pass all ranks (e.g., 0–7)
   legate_prof view /path/to/legate_*.prof

For more detail, see the official usage documentation for
``legate-profiler``.


Profiling cuPyNumeric Applications – Example 1
----------------------------------------------

Inefficient code
~~~~~~~~~~~~~~~~

.. code-block:: python

   import cupynumeric as np

   N = 10_000_000
   x = np.random.random(N).astype(np.float32)
   y = np.random.random(N).astype(np.float32)

   # advanced indexing, extra comms/overhead
   cond_idx = np.nonzero((x < 0.25) & (y > 0.5))

   # extra temporaries
   z     = x + y
   z_alt = x * y + 1.0

   # scatter writeback through indices (slower than boolean mask)
   z[cond_idx] = z_alt[cond_idx]

   # tiny chunked updates, lots of tiny tasks
   CHUNK = 4096
   for s in range(0, N, CHUNK):
       sub = z[s:s+CHUNK]
       gt1 = sub > 1.0
       sub[gt1]  = sub[gt1]  + 2.0
       sub[~gt1] = sub[~gt1] - 2.0

How this code works
~~~~~~~~~~~~~~~~~~~

This script builds two large random arrays, forms ``z = x + y``, then
selectively overwrites elements of ``z`` with ``x * y + 1.0`` where
``(x < 0.25) & (y > 0.5)``, and finally adjusts values above and below 1.0
by ±2.0. The performance suffers for three core reasons:

1. It uses ``nonzero(...)`` to create large integer index arrays and then
   scatters values back into ``z``, which adds metadata handling and
   communication overhead compared with simple boolean masks.
2. It creates extra temporaries (``x + y`` and ``x * y + 1.0``) instead of
   writing results into a preallocated output, increasing memory traffic and
   allocations.
3. It processes the array in 4,096-element slices, creating thousands of tiny
   tasks. The runtime spends a disproportionate amount of time scheduling and
   synchronizing rather than executing useful work.

These choices increase memory pressure, task-launch overhead, and communication
costs, making the computation scale poorly compared to a more direct,
vectorized approach.

Array creation
^^^^^^^^^^^^^^

.. code-block:: python

   x = np.random.random(N).astype(np.float32)
   y = np.random.random(N).astype(np.float32)

``np.random.random(N)`` returns float64 and then ``astype(np.float32)``
forces an extra copy instead of producing the target data type directly.

Index selection via ``nonzero``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   cond_idx = np.nonzero((x < 0.25) & (y > 0.5))

This builds index arrays with ``nonzero`` and forces a scatter write, increasing
memory use and kernel/communication overhead compared to a single, contiguous
masked update.

Temporaries
^^^^^^^^^^^

.. code-block:: python

   z     = x + y
   z_alt = x * y + 1.0

This section creates two large temporaries, increasing memory traffic and
allocations.

Scatter assignment
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   z[cond_idx] = z_alt[cond_idx]

The code scatters values back into ``z`` using advanced indexing, adding
overhead compared to mask-based updates.

Tiny-chunk loop
^^^^^^^^^^^^^^^

.. code-block:: python

   CHUNK = 4096
   for s in range(0, N, CHUNK):
       sub = z[s:s+CHUNK]
       gt1 = sub > 1.0
       sub[gt1]  = sub[gt1]  + 2.0
       sub[~gt1] = sub[~gt1] - 2.0

The loop breaks the array into thousands of small slices, which results in many
tiny tasks; runtime overhead dominates useful computation.

Profiler Output – Inefficient CPU Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CPU lane
^^^^^^^^

.. figure:: ../_images/profiling_debugging/cpu_inefficient.png
   :alt: Inefficient CPU profiler timeline with many tiny tasks
   :width: 90%

   CPU profiler view for the inefficient version. The "barcode" pattern of
   thin slivers represents thousands of tiny tasks instead of a few large ones.

The CPU lanes show a spike during the initial large element-wise operations
(e.g., ``z = x + y``, ``z_alt = x * y + 1.0``), followed by long stretches of
low utilization. The 4,096-element slice loop fragments work into many small
tasks, producing the barcode-like pattern. This over-fragmentation increases
scheduling and launch overhead, creates idle gaps, and lowers CPU efficiency.

Utility lane
^^^^^^^^^^^^

.. figure:: ../_images/profiling_debugging/utility_inefficient.png
   :alt: Inefficient utility lane with sustained meta-task load
   :width: 90%

   Utility lane for the inefficient version, showing sustained runtime
   overhead while computation is fragmented.

The utility lanes show sustained high activity almost the entire run, with only
a late drop. The runtime is continuously mapping, performing dependency
analysis, and launching thousands of micro-tasks created by the slice loop and
scatter pattern. Time is spent orchestrating rather than computing, often
coinciding with idle gaps in the CPU lanes.

I/O lane
^^^^^^^^

.. figure:: ../_images/profiling_debugging/io_inefficient.png
   :alt: Inefficient I/O lane with scattered top-level activity
   :width: 90%

   I/O lane for the inefficient version, with a long, low baseline of small
   transfers and coordination.

The I/O lane shows early heavy activity due to big copies, then a long low
baseline of small transfers, followed by a plateau near the end as outstanding
copies drain and instances are finalized. More time is going to data movement
and top-level coordination instead of compute.

System and Channel lanes
^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ../_images/profiling_debugging/channel_inefficient.png
   :alt: Inefficient Channel lane with many micro-copies
   :width: 90%

   Channel lane for the inefficient version. Gray bands compact many small
   DMA copies, another sign of over-granularity.

System activity stays relatively low and flat (mostly startup/shutdown), so it
is not the bottleneck. The Channel lanes, however, show an early large copy,
then a long, faint baseline of constant small scatter/gather copies. When
zoomed in, this resolves to hundreds or thousands of individual micro-copies,
each paying setup and synchronization overhead.

Efficient code
~~~~~~~~~~~~~~

.. code-block:: python

   import cupynumeric as np

   N = 10_000_000
   rng = np.random.default_rng()
   x = rng.random(N, dtype=np.float32)
   y = rng.random(N, dtype=np.float32)

   # In-place sum without a temporary
   z = np.empty_like(x)
   np.add(x, y, out=z)

   # Conditional overwrite with a boolean mask (faster than nonzero + scatter)
   cond = (x < 0.25) & (y > 0.5)
   np.putmask(z, cond, x * y + 1.0)

   # Wide masked updates, in-place
   gt1 = z > 1.0
   z[gt1]  += 2.0
   z[~gt1] -= 2.0

How this code works
~~~~~~~~~~~~~~~~~~~

This program generates two large ``float32`` arrays directly from the generator
API (no extra casts), computes ``z = x + y`` directly into a preallocated
output, selectively overwrites elements of ``z`` with ``x * y + 1.0`` where
``(x < 0.25) & (y > 0.5)``, and then applies two wide, in-place updates that
add or subtract 2.0 based on whether values exceed 1.0.

It is efficient because it:

- Avoids unnecessary temporaries by writing into a preallocated array.
- Uses a boolean mask instead of creating large index arrays.
- Performs the final adjustments as wide vectorized operations rather than many
  small slices.

These choices reduce memory traffic, task-launch overhead, and communication
costs, leading to better utilization and scalability on both CPU and GPU.

Efficient CPU profiler results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../_images/profiling_debugging/cpu_efficient.png
   :alt: Efficient CPU profiler timeline with few long tasks
   :width: 90%

   CPU profiler view for the efficient version. Work is consolidated into a
   few long tasks with high sustained utilization.

- Few longer bars and minimal "barcode" indicate that work is consolidated into
  large tasks; cores stay busy with little orchestration.
- Large, contiguous vector ops (add, mask, updates) keep per-task overhead tiny
  versus compute.

Utility, I/O, and Channel lanes (efficient)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../_images/profiling_debugging/utility_efficient.png
   :alt: Efficient utility lane with short bursts
   :width: 90%

   Utility lane for the efficient version, with quiet baseline and brief bursts.

.. figure:: ../_images/profiling_debugging/io_efficient.png
   :alt: Efficient I/O lane with minimal top-level overhead
   :width: 90%

   I/O lane for the efficient version, dominated by a single top-level block.

.. figure:: ../_images/profiling_debugging/channel_efficient.png
   :alt: Efficient Channel lane with a few bulk transfers
   :width: 90%

   Channel lane for the efficient version, showing a few bulk transfers and no
   persistent small-copy baseline.

Utility shows a quiet baseline with brief bursts at startup and teardown.
Mapping and scheduling are compact; most time goes to real compute. The I/O
lane is dominated by a single top-level task with a few short blips for
init/teardown. The Channel lane shows a handful of short, tall bursts (bulk
transfers), with no thin, persistent copy baseline.

Efficient multi-GPU results
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../_images/profiling_debugging/gpu_efficient_all_ranks.png
   :alt: Efficient multi-GPU profiler view across ranks
   :width: 90%

   Multi-GPU profiler view for the efficient version, aggregated across ranks.

The GPU Dev lanes show steady compute time: the average line is high and stays
high while work runs, with few wide kernels and short gaps. The GPU Host lanes
show mostly quiet baselines with short bursts aligned to device kernels,
indicating minimal launch/orchestration overhead.


Profilers – Wrap Up
-------------------

By using Legate’s built-in profiler, you gain the ability to uncover hidden
bottlenecks and inefficiencies in your code. Profiling does not just expose
bugs; it provides a lens to reason about performance and systematically improve
it. Small structural tweaks (fusing operations, avoiding scatter writes, and
cutting temporaries) translate into fewer tasks, less orchestration, and higher
throughput. This is the difference between code that “just runs” and code that
is efficient, scalable, and production-ready.


Understanding and Handling Out-of-Memory (OOM) Issues – Example 2
------------------------------------------------------------------

How OOM occurs
~~~~~~~~~~~~~~

cuPyNumeric runs on top of Legate Core. At launch, the ``legate`` launcher
auto-sizes memory pools for each “memory kind” it detects (e.g., CPU
``SYSTEM_MEM``, GPU framebuffer) on the assigned process/GPU. You can override
these defaults to fixed sizes if needed with flags such as ``--sysmem`` (MiB of
host DRAM) and ``--fbmem`` (MiB of GPU memory). If an operation needs to create
a new instance that exceeds the reserved capacity of a pool, the runtime raises
an out-of-memory error for that memory kind (e.g., ``SYSTEM_MEM`` or
``FBMEM``) and reports the task/store that triggered it.

Most “mystery OOMs” are not total node exhaustion; they are
per-process, per-kind pool exhaustion. The fix is often to:

1. Right-size those pools.
2. Reduce peak live instances so they fit.

For more information on Legate Core, see the core overview documentation.

Demo script
~~~~~~~~~~~

.. code-block:: python

   import cupynumeric as np

   # allocation site, not instantiated yet
   a = np.ones((1024 * 1024 - 2,))

   # allocation site, not instantiated yet
   b = np.zeros((1024 * 1024,))

   # use only a slice of b; causes b#1
   b[1:-1] = a

   # use full b; causes instance b#2
   c = b + 2

   # will fail
   d = c + 3

CPU-only run (deterministic OOM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # LEGATE_TEST=1: verbose allocation diagnostics
   LEGATE_TEST=1 legate --cpus 1 --gpus 0 --sysmem 40 oom.py

``LEGATE_TEST=1`` enables verbose diagnostics: detailed allocation information
such as logical store creation, instance sizes, and memory reservations.

``legate --cpus 1 --gpus 0 --sysmem 40 oom.py`` runs the script with one CPU
worker and a fixed system memory pool of 40 MiB. Any time an operation requires
more than this reserved pool, you will see a ``Failed to allocate ... of kind
SYSTEM_MEM`` error.

GPU run (framebuffer behavior)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Single GPU, intentionally tight framebuffer pool
   LEGATE_TEST=1 legate --cpus 2 --gpus 1 --fbmem 40 --sysmem 512 oom.py

.. note::

   Flags are per process. If you use multiple ranks per node, each rank needs
   its own slice of ``--sysmem`` and ``--fbmem``.

Step 1 – Read the OOM line
~~~~~~~~~~~~~~~~~~~~~~~~~~

When an OOM happens, the failure line tells you:

- The memory kind that ran out (e.g., ``SYSTEM_MEM`` or ``FBMEM``).
- Which task/logical store was being created when it failed.

That points you at the operation that spiked usage.

.. figure:: ../_images/profiling_debugging/oom_error.png
   :alt: Example OOM error message in the Legate runtime
   :width: 90%

   Example OOM error for the demo script, highlighting the memory kind and
   source location.

The important failure line looks like:

.. code-block:: text

   Failed to allocate 8388608 bytes on memory 1e00000000000000 (of kind SYSTEM_MEM)
   for region requirement(s) {1} of Task cupynumeric::BinaryOpTask[/home/USER/d/cupynumeric/oom.py:16] (UID 8)

This means:

- Legate attempted to allocate an 8 MiB array in the 40 MiB ``SYSTEM_MEM``
  pool for the ``BinaryOpTask`` at line 16.
- No contiguous free block was available.
- The OOM originates from the operation ``d = c + 3`` in the demo.

The rest of the error message lists existing logical stores, their sizes, and
the lines that created them. From this, you can see that previous arrays and
allocations (``a``, ``b``, ``c``) occupy roughly 32 MiB of the 40 MiB pool,
and the remaining space is not enough once allocator overhead and alignment
are included.

Step 2 – Verify resource reservations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Confirm the runtime actually reserved the memory you expect using
``--show-config``. Remember that flags are per process; with multiple ranks
per node, each process needs its own slice of CPU/GPU memory.

.. code-block:: bash

   # Print the pools you would use...
   legate --cpus 1 --gpus 0 --sysmem 40 --show-config \
   && LEGATE_TEST=1 legate --cpus 1 --gpus 0 --sysmem 40 oom.py

.. figure:: ../_images/profiling_debugging/show_config.png
   :alt: Example legate --show-config output
   :width: 90%

   Example ``legate --show-config`` output for a CPU-only run.

This catches misconfiguration, confirms per-rank ``--sysmem``/``--fbmem``/``--zcmem``
match your intent, and gives a one-line snapshot to paste into bug reports.

Step 3 – Sanity-check device memory externally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On GPU nodes, also glance at external tools to confirm there is headroom on the
host and each selected device.

.. code-block:: bash

   # CPU (host RAM)
   cat /proc/meminfo | grep MemAvailable

   # GPU (device VRAM)
   nvidia-smi

.. figure:: ../_images/profiling_debugging/meminfo.png
   :alt: Example MemAvailable output
   :width: 70%

.. figure:: ../_images/profiling_debugging/nvidia_smi.png
   :alt: Example nvidia-smi output
   :width: 70%

Mitigation strategies
---------------------

Depending on the root cause you identified from the OOM message and diagnostics,
you can use several mitigation strategies. Most real workloads benefit from a
combination of these rather than a single fix.

A. Resize Legate’s memory reservations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, Legate uses all available per-rank memory kinds (``SYSTEM_MEM``,
``FBMEM``, ``ZCMEM``) unless you constrain them. Use ``--sysmem`` /
``--fbmem`` (and optionally ``--zcmem``) to size pools explicitly.

When to **increase** memory:

- OOM cites “of kind SYSTEM_MEM / FBMEM”, and external tools show free memory.
- You intentionally keep multiple large arrays/live instances.
- You have already minimized temporaries/scatter/tiny tasks but still hit pool
  limits.

When to **decrease** memory:

- Reservation fails at startup (cannot pre-reserve).
- Many ranks per node; ``R × --sysmem`` / ``R × --fbmem`` would exceed capacity.
- You want less VRAM pinned or prefer host placement/offload.

Examples:

.. code-block:: bash

   # CPU-only (increase SYSTEM_MEM)
   LEGATE_TEST=1 legate --gpus 0 --cpus 1 --sysmem 128 oom.py

   # Single-GPU (tight but sane pools; allow host spill)
   LEGATE_TEST=1 legate --gpus 1 --cpus 2 --fbmem 128 --sysmem 512 oom.py

B. Prefetch the data
~~~~~~~~~~~~~~~~~~~~

Prefetching fetches data into memory before heavy compute, avoiding duplicate
instances and peak spikes.

Technique 1 – ``stencil_hint`` for halos
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import cupynumeric as np

   N = 1024 * 1024
   a = np.ones((N - 2,), dtype=np.float32)
   b = np.zeros((N,),      dtype=np.float32)

   # We will touch b[1:-1] now and later its full range → halo 1 on each side
   b.stencil_hint(low_offsets=(1,), high_offsets=(1,))

   b[1:-1] = a
   c = b + 2
   d = c + 3

Calling ``stencil_hint`` allocates a larger backing instance up front (including
halo) so downstream ops reuse it and do not grow the instance mid-band.

Technique 2 – Whole-array touch with ``out=``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import cupynumeric as np

   N = 1024 * 1024
   a = np.ones((N - 2,), dtype=np.float32)
   b = np.zeros((N,), dtype=np.float32)

   # GPU: materializes in FBMEM; CPU-only: materializes in SYSTEM_MEM
   np.multiply(b, 1, out=b)      # or: np.add(b, 0, out=b) or: b *= 1

   b[1:-1] = a
   c = b + 2
   d = c + 3

Using ``out=`` (or in-place) guarantees no second full-size temporary is
created while you prefetch, and forces materialization on the target memory
without extra allocations.

C. Releasing memory between phases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Drop references, collect garbage, and flush allocator caches to shrink the live
working set so the next phase has headroom.

.. code-block:: python

   # 1) Drop references
   big = None
   if "big" in globals():
       del big
   cache.clear()  # if you stored big arrays in dicts/lists/closures

   # 2) Reclaim cyclic garbage
   import gc
   gc.collect()

   # 3) If you used CuPy, flush its pools
   try:
       import cupy as cp
       cp.get_default_memory_pool().free_all_blocks()
       cp.get_default_pinned_memory_pool().free_all_blocks()
   except Exception:
       pass

D. Offload to CPU memory
~~~~~~~~~~~~~~~~~~~~~~~~

If you have optimized code and pools but still need more GPU space, offload
arrays to CPU memory.

.. code-block:: python

   import cupynumeric as np
   from legate.core import StoreTarget
   from legate.core.data_interface import offload_to

   N = 1024 * 1024
   a = np.ones((N - 2,), dtype=np.float32)
   b = np.zeros((N,), dtype=np.float32)

   # Work that builds pressure in FBMEM
   b[1:-1] = a
   c = b + 2

   # Offload 'c' to host RAM
   offload_to(c, target=StoreTarget.SYSMEM)

   d = c + 3

``offload_to`` copies an array to the target memory (e.g., system RAM) and
discards other copies (e.g., in GPU framebuffer), immediately freeing VRAM for
later GPU work.

E. Coding practices to reduce peak memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If mitigations above are not enough, reduce the peak working set by:

- Avoiding large index arrays and scatter writes; use boolean masks instead of
  ``nonzero(...)`` + advanced indexing.
- Avoiding per-slice “tiny task” loops on big arrays; use whole-array
  vectorized operations instead.
- Avoiding unnecessary temporaries; write into preallocated outputs using
  ``out=`` when possible.

The Legate profiler can help you spot phases where Framebuffer/System lanes
ramp up and stay high, Channel shows persistent baselines, and Utility shows
“confetti” of meta tasks—all signs of duplicated instances and peak usage.

Last resort: downcast to a smaller data type (e.g., ``float32`` instead of
``float64``) when numerically acceptable. Prefer mixed precision if full
downcasting is risky.


Putting it together – Example mitigations
-----------------------------------------

Returning to the earlier OOM demo:

.. code-block:: python

   import cupynumeric as np

   a = np.ones((1024 * 1024 - 2,))
   b = np.zeros((1024 * 1024,))

   b[1:-1] = a
   c = b + 2
   d = c + 3

You observed:

- Step 1 (read the OOM line): the failure is pool exhaustion; instances from
  ``a``, ``b``, and ``c`` are still live, so ``d = c + 3`` cannot fit.
- Step 2 (verify resource reservations): pool sizes match flags.
- Step 3 (sanity-check headroom): host/GPU still have capacity, so the issue is
  the per-process pool and peak live instances, not total node memory.

Possible fix paths:

- **Path A – Resize per-rank pools** (mitigation A)  
  Increase ``--sysmem`` / ``--fbmem`` as capacity allows.

- **Path B – Prefetch using whole-array touch** (mitigation B)  
  Use a no-op ``out=`` ufunc to materialize the full array up front, avoiding
  mid-compute growth.

- **Path C – Reduce the live working set** (mitigation C)  
  Drop references and use in-place operations instead of creating new arrays.

Quick re-checks before rerun:

.. code-block:: bash

   legate --show-config                  # pools per rank match intent?
   cat /proc/meminfo | grep MemAvailable # host headroom
   nvidia-smi                            # GPU headroom

Result: Path A expands the pool; Paths B and C reduce peak usage. Either
resolves this example OOM for ``d = c + 3`` and makes the cause and fix
explicit.

Important: These strategies are shown on a simplified example. For more complex
pipelines, also consider offloading and structural code changes alongside these
techniques.
