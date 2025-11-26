Overview
--------

This section assumes familiarity with running cuPyNumeric, extending it with
Legate Task, and scaling gradient boosting with Legate Boost. For a refresher,
see:

- Setting up your environment and running cuPyNumeric
- Extending cuPyNumeric with Legate Task
- Scaling gradient boosting with Legate Boost

cuPyNumeric scales familiar NumPy workloads seamlessly across CPUs, GPUs, and
multi-node clusters. Previous sections covered how to get code running; here
the focus shifts to making workloads production-ready. At scale, success is
not just about adding GPUs or nodes, it requires ensuring that applications
remain efficient, stable, and resilient under load. That means finding
bottlenecks, managing memory effectively, and preventing failures before
they disrupt a job.

This section focuses on two advanced capabilities in cuPyNumeric and the
Legate runtime that address these challenges:

- Profiling cuPyNumeric applications – to tune performance and analyze
  scalability. Profiling reveals bottlenecks such as idle GPUs,
  synchronization delays, or overly fine-grained tasks, helping you restructure
  code for better scaling.
- Debugging and Out-of-Memory (OOM) strategies – to improve reliability
  in memory-intensive workloads. These tools help diagnose crashes, manage
  GPU/CPU memory effectively, and prevent common anti-patterns so applications
  remain robust under heavy loads.

What you’ll gain: By combining profiling tools with solid OOM-handling
strategies, you can significantly improve the efficiency, scalability,
and reliability of cuPyNumeric applications across CPUs, GPUs, and
multi-node systems.

For more detail, see the official references:

- cuPyNumeric Documentation
- Legate Documentation


Usage
-----

1) To install the built-in Legate profiler tool in your Conda environment, run:

.. code-block:: bash

   conda install -c conda-forge -c legate legate-profiler

2) After installing the Legate profiler (legate-profiler), profile the code
using the ``--profile`` flag:

.. code-block:: bash

   # CPU example
   legate --cpus 8 --sysmem 4000 --profile myprog.py

   # Single GPU example
   legate --gpus 1 --profile myprog.py

   # Multi-GPU example (single node, multi-rank: 4 ranks × 1 GPU)
   srun -n 4 --mpi=pmix legate --gpus 1 --profile myprog.py

   # Multi-node example (2 nodes × 4 GPUs = 8 ranks × 1 GPU)
   srun -N 2 --ntasks-per-node=4 \
        --gpus-per-task=1 --gpu-bind=single:1 \
        --mpi=pmix -C gpu \
     legate --gpus 1 --profile myprog.py

3) Similarly, a program can be run via the ``LEGATE_CONFIG`` environment
variable:

.. code-block:: bash

   LEGATE_CONFIG="--cpus 8 --sysmem 4000 --profile" python ./myprog.py

4) After a run completes, in the directory you ran the command you’ll see:

- A folder: ``legate_prof/``, a self-contained HTML report
  (open ``legate_prof/index.html``)
- One or more raw trace files: ``legate_*.prof`` (one per rank)

The ``legate_*.prof`` files are what you need to view locally on your
machine.

Examples:

.. code-block:: text

   # CPU / Single GPU - Will only produce 1 file (1 rank)
   legate_0.prof
   legate_prof/

   # Multi-GPU - Will produce 1 file per rank (4 ranks)
   legate_0.prof
   legate_1.prof
   legate_2.prof
   legate_3.prof
   legate_prof/

   # Multi-Node (multi-rank; e.g, 2 nodes x 4 GPUs = 8 ranks)
   legate_0.prof ... legate_7.prof
   legate_prof/

.. note::

   Trace files are numbered by rank index (e.g., ``legate_0.prof``,
   ``legate_1.prof``), not by run. If you run again in the same directory,
   files with the same rank numbers will be overwritten; for example, a
   1-rank run will replace ``legate_0.prof``. The ``legate_prof/`` HTML
   report directory is also overwritten.

5) Local Setup: WSL, Miniforge (Conda), and Legate + Legate profiler viewer

These commands will work directly on a Linux environment.

For Windows OS – open Ubuntu/WSL2 (Windows Subsystem for Linux), install
Miniforge (Conda), and activate it.

.. code-block:: bash

   # Download installer
   wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

   # Install into home
   bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"

   # Load Conda into current shell
   source "$HOME/miniforge3/etc/profile.d/conda.sh"

   # Configure future shells
   conda init bash

   # Restart shell to apply changes
   exec $SHELL -l

   # Create and activate an environment with Legate + cuPyNumeric + the profile viewer
   conda create -n legate -y -c conda-forge -c legate legate cupynumeric legate-profiler
   conda activate legate

6) Copy files to your local device: Create a single top-level folder & keep
runs separated to avoid name/file clashes:

.. code-block:: bash

   # Copy legate_*.prof file(s) for CPU, single-GPU, Multi-GPU, or Multi-Node
   scp -r <USER>@<REMOTE_HOST>:<REMOTE_RUN_DIR>/legate_*.prof \
         "<LOCAL_DIR>/<FOLDER_NAME>/name_of_run"

7) In local machine, use the following command to open files with the profile
viewer:

.. code-block:: bash

   # CPU/GPU: single file (rank 0/N0)
   legate_prof view /path/to/legate_0.prof

   # Multi-GPU/Multi-Node: multiple ranks (pass them all: e.g: N0, N1, N2, etc)
   legate_prof view /path/to/legate_*.prof

For more detail, see the official references:

- Usage — NVIDIA legate


Profiling cuPyNumeric Applications with Legate Profilers – Example 1
--------------------------------------------------------------------

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
``(x < 0.25) & (y > 0.5)``, and finally inserts values above and below 1.0
by ±2.0. The performance suffers for three core reasons. First, it uses
``nonzero(...)`` to create large integer index arrays and then scatters values
back into ``z``, which adds metadata handling and communication overhead
compared with simple boolean masks. Second, it creates extra temporaries
(``x + y`` and ``x * y + 1.0``) instead of writing results into a preallocated
output, increasing memory traffic and allocations. Third, it processes the
array in 4,096-element slices, creating thousands of tiny tasks; the runtime
spends a disproportionate amount of time scheduling and synchronizing rather
than executing useful work. These choices increase memory pressure,
task-launch overhead, and communication costs, making the computation scale
poorly compared to a more direct, vectorized approach.

Array creation
^^^^^^^^^^^^^^

.. code-block:: python

   x = np.random.random(N).astype(np.float32)
   y = np.random.random(N).astype(np.float32)

The snippet generates two large input arrays but uses ``.astype(...)``, which
forces an extra copy instead of producing the target data type directly.
``np.random.random(N)`` returns ``float64``, and produces an array of length
``N`` filled with random floats sampled from ``[0, 1)``. The
``.astype(np.float32)`` converts it to single precision (``float32``), which
halves the memory footprint.

Index selection via ``nonzero``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   cond_idx = np.nonzero((x < 0.25) & (y > 0.5))

Here the code builds index arrays with ``nonzero``. ``nonzero`` builds large
index arrays and forces a scatter write, increasing memory use and
kernel/communication overhead compared to a single, contiguous masked update.

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

Now the code scatters values back into ``z`` using advanced indexing, adding
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

Finally the code breaks the array into thousands of small slices, which results
in many tiny tasks; runtime overhead dominates useful computation.

Profiler Output and Interpretation - Inefficient CPU Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1) CPU
^^^^^^^

.. image:: ../_images/profiling_debugging/cpu_inefficient.png
   :alt: Inefficient CPU profiler timeline with many tiny tasks
   :width: 90%

What this shows
"""""""""""""""

User’s compute tasks, computations, and data movement on main CPU worker
cores. Long, solid bars means a few large tasks/operations (good). Dense
“bar-code” slivers means many tiny tasks (bad). This is where you read task
time and spot idle gaps between tasks.

**Zoomed in:**

CPU Observation:
Start-up shows a few large initialization tasks. After that, the 4,096-element
slice loop fragments work into many small tasks, producing a barcode-like
pattern. This displays overly fine-grained work that increases
scheduling/launch overhead, creates idle gaps, and lowers CPU efficiency. Each
small sliver represents “merged tasks” that execute separate small
computations.

CPU Avg
"""""""

We observe a sharp startup spike in the CPU average line (~74% utilization),
followed by a long, low plateau. The spike corresponds to the large
element-wise operations (``z = x + y``, ``z_alt = x*y + 1.0``). The subsequent
flat, low amplitude reflects the ``CHUNK = 4096`` loop breaking work into
thousands of short slices, so cores never fully saturate. In the worker lanes
(c2-c9), this appears as a few early long, dark bars for the big operations,
then dense “bar-code” slivers across many cores for the rest of the run. Each
sliver is a tiny task from the slice loop. This fragmentation is bad: it
reduces sustained CPU utilization, increases context switching, and hurts cache
locality, so time shifts from steady computation to orchestrating tiny tasks.

What would good look like? A handful of long bars for the main operations,
then just two long masked updates (plus/minus 2.0 step), as opposed to
thousands of slivers.

2) Utility
^^^^^^^^^^^

.. image:: ../_images/profiling_debugging/utility_inefficient.png
   :alt: Inefficient utility lane with sustained meta-task load
   :width: 90%

What this shows
"""""""""""""""

Legate runtime “meta” work: dependence analysis, mapping, task launch, and
coordination. These are tasks needed for the library to function but are not
the user’s computation. You want short bursts around big tasks/operations.
A sustained plateau means the scheduler is the bottleneck (threads are waiting
on runtime work).

Utility Observation:
Sustained high activity almost the entire run, with only a late drop, runtime
overhead dominates while on the other hand the computation is fragmented.

Utility Avg
"""""""""""

We observe a quick ramp-up into a long, flat plateau on the utility-average
line, followed by a drop near the end. The plateau indicates the runtime is
continuously mapping, performing dependency analysis, and launching thousands
of micro-tasks created by the ``CHUNK = 4096`` slice loop. Near the tail, the
utility load decreases because launches are over and only cleanup/final copies
remain. In the utility lanes (u0-u1), this appears as dense “confetti” of tiny
meta-tasks, which is the signature of over-granularity keeping the scheduler
busy almost all the time. The slice loop (``for s in range(0, N, 4096)``)
drives persistent mapping/launch work; the index selection + scatter pattern
(``nonzero`` + ``z[idx] = …``) adds per-slice dependence checks and
data-placement decisions; and extra temporaries (``z``, ``z_alt``) create more
instances for the runtime to allocate and track. Bottom line: bad, time is
more so spent orchestrating rather than computing, often coinciding with idle
gaps on the CPU lanes.

What would good look like? Short, discrete bursts around a few large
tasks/operations (in-place add, one masked overwrite, two whole-array
threshold updates), with the utility lanes mostly quiet between them.

3) I/O (input/output)
^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../_images/profiling_debugging/io_inefficient.png
   :alt: Inefficient I/O lane with scattered top-level activity
   :width: 90%

What this shows
"""""""""""""""

This lane is Legate ``TopLevelTask`` / driver time; file reads/writes are a
subset. Use it alongside Channel (which records data between host and device)
to reason about data movement. Spikes usually reflect large reads/writes or
heavy top-level coordination; a thin, steady baseline suggests many small
I/O/driver events. GPU or CPU gaps often correlate with I/O or Channel
activity, but can also come from Utility (mapping) or dependencies.

I/O observation:
We see early heavy activity due to big copies, then a long low baseline of
small transfers, followed by a tall plateau near the end. The large data
movement pattern itself comes through Channel (scatter writes + tiny chunks),
while the I/O lane shows the top-level Python/driver work around it. The dark
magenta color represents I/O tasks actively executing. The lighter pink you
see earlier are the same I/O tasks while waiting/ready but not running yet.
The profiler will use shade to indicate state:

- Darkest shade = actively executing
- Intermediate shade = ready state
- Lightest shade = task is blocked
- Gray = groups of tiny tasks

I/O Avg
"""""""

We observe an early spike that settles into a short plateau as the program
initializes and writes full arrays for the temporaries (``z = x + y``,
``z_alt = x*y + 1.0``). This transitions into a long, low baseline reflecting
ongoing top-level coordination associated with many small, non-contiguous
transfers triggered by the ``nonzero`` scatter and the ``CHUNK = 4096`` slice
loop (you’ll see the transfer shape itself in Channel as a thin, persistent
baseline). Near the end, the line rises again, a late plateau, as outstanding
copies drain and instances are finalized during cleanup, then it drops to
zero. Bottom line: Bad, more time is going to data movement/coordination
instead of compute, and it correlates with high Utility and fragmented CPU
(idle/long-poll symptoms).

What would good look like? Brief I/O bursts only: write once to a preallocated
output, one masked overwrite, then quiet channels, few wide transfers, no long
baseline.

4) System
^^^^^^^^^^

What this shows
"""""""""""""""

Low-level system memory activity: allocations, thread/process setup, OS
interaction, and other background work. It should be quiet and flat during
steady computation.

System Observation:
We observe a small early bump from normal startup work (process/thread creation
and initial allocation), followed by a flat, low sitting plateau indicating
minimal ongoing system overhead, and a slight dip at the end as the program
shuts down and cleans up. Bottom line: good/neutral, system stays low and
stable; the real bottlenecks are elsewhere (CPU fragmentation, Utility
overhead, and I/O/Channel traffic).

5) Channel (chan)
^^^^^^^^^^^^^^^^^^

.. image:: ../_images/profiling_debugging/channel_inefficient.png
   :alt: Inefficient Channel lane with many micro-copies
   :width: 90%

What this shows
"""""""""""""""

Communication pathways grouped by source to destination memory. These are the
Direct Memory Access (DMA) copies that move data between host DRAM, GPU
Framebuffer memory, zero-copy memory, and sometimes system/network buffers.
Tall wide bursts mean large continuous transfers (efficient). Thin, persistent
baselines means many small transfers, often from over-granular work
(scatter/gather, slice loops).

Channel Observation:
The gray rectangles are merged visuals: when zoomed out, the profiler compacts
hundreds/thousands of micro-copies into gray bands; zooming in reveals the
individual narrow copy boxes, confirming over-granularity. Net effect: poor
effective throughput, each micro-copy pays setup/latency and increases
mapping/synchronization load (seen as a busy Utility lane), and the fragmented
transfers create idle gaps on CPU lanes while tasks wait for data.

Channel Avg
"""""""""""

One early blip (initial large copy into device memory), then a long, faint
baseline, which is the flood of small scatter/gather copies from
``nonzero(...)`` indexing and the ``CHUNK = 4096`` loop. Each tiny slice
forces its own DMA operation.

What would good look like? A handful of short, tall bursts around the major
steps: write once into ``z``, do a single masked overwrite for the condition,
then perform two whole-array threshold updates. Between bursts the baseline
stays quiet, utility lanes are mostly idle, and CPU lanes show long, solid
bars instead of bar-code slivers.

6) Dependent Partitioning (dp)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

What this shows
"""""""""""""""

Time the runtime spends creating dependent region partitions (subregions
derived from other regions), this is needed for sliced, indirect, or masked
operations. Partitioning must finish before mapping/copies can proceed. It’s a
niche, hyper-specific metric that often reads near zero in simple workloads,
but can matter in more complex applications (e.g., sparse matrix operations,
irregular graphs, adaptive meshes) where partition shapes change frequently.

DP Observation:
The dp avg line is flat at ~0% utilization for the whole run, indicating
partition work is negligible in total time. In dp0 you still see a long gray
“merged” band: that's the view compacting many ultra-short partition events
(from the 4,096-element slice loop and the ``nonzero(...)`` scatter) into a
single bar at this zoom level. Each event is tiny, so even though there are
lots of them, their duty cycle is so low that average utilization rounds to
zero.

Interpretation
""""""""""""""

Partition creation is not the primary bottleneck here (Utility and Channel
are). However, those micro-partitions still add overhead and can lengthen
critical paths by forcing extra dependencies before copies/updates launch. The
better alternative would be to keep dp light by avoiding per-slice/indirect
updates and preferring whole-array masked writes and reused partitions so the
runtime doesn't need to generate countless micro-partitions.


Inefficient GPU Results - (4 Ranks 1 GPU each)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All ranks:

.. image:: ../_images/profiling_debugging/gpu_inefficient_all_ranks.png
   :alt: Inefficient multi-GPU profiler view across ranks
   :width: 90%

1) GPU Dev
^^^^^^^^^^^

What this shows
"""""""""""""""

Execution of kernels directly on the GPU Device. This lane measures how long
GPU execution units are busy running element-wise operations, reductions,
matrix kernels, etc. High steady utilization means kernels are big and
well-batched; low or jagged utilization means the GPU is either idle or
getting too many tiny launches.

GPU Dev Observation:
In the GPU Dev lane we see wide gray bands at zoom compress many
micro-kernels; zooming in reveals dense strips, each a tiny kernel from the
4,096-element slices or the scatter path. Above that we see many repeated,
sawtooth-like spikes rather than long, solid bars. ``z = x + y`` and
``z_alt = x* y + 1.0`` each launch element-wise kernels early (brief higher
utilization). ``cond_idx = np.nonzero((x < 0.25) & (y > 0.5))`` computes a
boolean test and then materializes index arrays; subsequent
``z[cond_idx] = z_alt[cond_idx]`` performs a scatter update that splits work,
producing multiple small kernels instead of one contiguous masked write. The
loop with ``CHUNK = 4096`` (``sub = z[s:s+CHUNK]``; threshold; two per-slice
updates) generates thousands of tiny, per-chunk kernels. Each chunk does:
compare, select, then two updates, so the device keeps starting and stopping
kernels rather than running a few big ones.

GPU Dev Avg
"""""""""""

At startup the line lifts due to big element-wise operations. It then gradually
sinks lower, oscillating high to low. That indicates persistent GPU activity,
but fine granularity: per-chunk/per-scatter kernels are short, so launch
overhead and synchronization eats into total time.

What would good look like? A few long, contiguous kernels that keep the device
busy: one large vector add, one single masked overwrite (no scatter), etc. The
GPU Dev lane shows long solid bars with a high, steady average line, minimal
gaps between kernels, and compute overlapping cleanly with a few bulk copies
(seen in Channel).

2) GPU Host
^^^^^^^^^^^^

What this shows
"""""""""""""""

CPU-side orchestration for GPU work: kernel launches, argument setup,
enqueueing tasks, and prepping memory transfers. You want brief bursts per
large kernel, not continuous chatter/oscillation.

GPU Host Observation:
Frequent spikes/oscillations mirror GPU Dev, such that the code launches many
tiny kernels:

- ``nonzero(...)`` + ``z[cond_idx] = z_alt[cond_idx]`` adds scatter setup and
  extra small launches.
- ``CHUNK = 4096`` loop creates per-slice compare + two updates, so the host
  repeatedly launches micro-kernels.

GPU Host Avg
""""""""""""

High, jagged baseline after a startup spike means launch overhead is sustained.
Host time tracks GPU Dev closely, an obvious indication of over-granularity
(per-launch cost comparable to work done).

What would good look like? Sparse, short spikes only when launching those few
large kernels. The GPU Host lane is a little lower and quieter than GPU Dev.
Brief bursts at kernel starts, then long idle periods while the device
executes. No dense “barcode” of micro-launches. Will look very similar to GPU
Dev.

3) Zerocopy
^^^^^^^^^^^^

What this shows
"""""""""""""""

Transfers between CPU host memory and GPU memory using pinned host memory
directly accessible by the GPU. Useful when data is accessed only once or in
small pieces. Ideally, you see just a few bursts; heavy use usually means data
isn't staged efficiently in device memory.

Zerocopy Observation:
The avg line is pinned at a ~0% utilization for the entire run. If this section
was expanded you would see many blocks gradually getting larger as you scroll
down due to the 4,096-element slice loop, but their duty cycle is so small that
utilization rounds to zero. Any zero-copy use here is incidental and negligible
compared with other streams. Zerocopy is not a bottleneck.

4) Framebuffer
^^^^^^^^^^^^^^^

What this shows
"""""""""""""""

Time the profiler records GPU Framebuffer (device memory) allocation,
deallocation, or access overhead. This isn't the math itself, but the memory
management cost for storing temporaries and outputs in device memory. Ideally
this lane should stay low and quiet, with only brief bumps for allocation at
startup and cleanup at shutdown.

Framebuffer Observation:
The avg line rises gradually to ~2–3% utilization and holds steady through most
of the run, dipping only near the end. That reflects sustained
allocation/instance traffic, likely from:

- Extra temporaries (``z = x + y``, ``z_alt = x*y+1.0``) creating more device
  instances than necessary.
- Scatter updates (``z[cond_idx] = …``) forcing additional partitioned storage.
- The ``CHUNK = 4096`` loop repeatedly touching small subregions.

What would good look like? A small bump at initialization (allocate main
arrays), flat near zero during steady compute, and a dip at the end (cleanup).
No continuous Framebuffer overhead, just data living in device memory for long
stretches while kernels run.


Efficient Code
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

This program generates two large ``float32`` arrays directly from the Generator
API (no extra casts), computes ``z = x + y`` directly into a preallocated
output, selectively overwrites elements of ``z`` with ``x*y + 1.0`` where
``(x < 0.25) & (y > 0.5)``, and then applies two wide, in-place updates that
add or subtract 2.0 based on whether values exceed 1.0. It’s efficient because
it avoids unnecessary temporaries by writing into a preallocated array, uses a
boolean mask instead of creating index arrays, and performs the final
adjustments as wide vectorized operations rather than many small slices. These
choices reduce memory traffic, task-launch overhead, and communication costs,
leading to better utilization and scalability on both CPU and GPU.

Array creation (data type & copies)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # inefficient
   x = np.random.random(N).astype(np.float32)
   y = np.random.random(N).astype(np.float32)

   # efficient
   rng = np.random.default_rng()
   x = rng.random(N, dtype=np.float32)
   y = rng.random(N, dtype=np.float32)

Using the Generator API, ``rng = np.random.default_rng(); x = rng.random(N,
dtype=np.float32)``, creates arrays in the target dtype at the source, so
there’s no ``float64`` to ``float32`` downcast and no extra allocation/copy.
That cuts memory traffic and peak footprint, improves cache/GPU memory
efficiency, and avoids the false impression that ``astype(copy=False)`` would
help, since casting to a new dtype always requires a copy. In short: fewer
bytes moved, fewer temporaries, faster start-up.

Base computation
^^^^^^^^^^^^^^^^

.. code-block:: python

   # inefficient
   z = x + y

   # efficient
   z = np.empty_like(x)
   np.add(x, y, out=z)

Both compute ``x + y``. The efficient code writes directly into a preallocated
output, avoiding a full temporary allocation and an extra pass over memory.
This reduces peak memory and improves cache/GPU memory efficiency.

Conditional overwrite (indices vs mask)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # inefficient
   cond_idx = np.nonzero((x < 0.25) & (y > 0.5))
   z_alt = x * y + 1.0
   z[cond_idx] = z_alt[cond_idx]

   # efficient
   np.putmask(z, (x < 0.25) & (y > 0.5), x * y + 1.0)

The efficient version applies the condition directly inside ``putmask``, so no
large index arrays are built and it is easier for the runtime to fuse/optimize.
This keeps the update lightweight and communication-friendly. The inefficient
version materializes index arrays, creates an additional full-size temporary
(``z_alt``), and performs a scatter assignment, each adding overhead.

Chunked loop vs Vectorized
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # inefficient
   CHUNK = 4096
   for s in range(0, N, CHUNK):
       sub = z[s:s+CHUNK]
       gt1 = sub > 1.0
       sub[gt1]  = sub[gt1]  + 2.0
       sub[~gt1] = sub[~gt1] - 2.0

   # efficient
   gt1 = z > 1.0
   z[gt1]  += 2.0
   z[~gt1] -= 2.0

The efficient approach performs two wide, in-place vectorized updates over the
whole array. This eliminates thousands of tiny tasks, dramatically reducing
launch and scheduling overhead and improving GPU/CPU utilization. The more
regular access pattern also plays nicely with caches and the memory controller,
boosting overall utilization.

Profiler Output and Interpretation - Efficient CPU Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CPU
^^^

.. image:: ../_images/profiling_debugging/cpu_efficient.png
   :alt: Efficient CPU profiler timeline with few long tasks
   :width: 90%

Why this is good:
Few longer bars, minimal “barcode.” Work is consolidated into large tasks;
cores stay busy with little orchestration. CPU avg: Large, contiguous vector
ops (add, mask, updates) keep per-task overhead tiny vs compute, so the
runtime batches the work rather than slicing it into thousands of tiny tasks.

Efficient Code:

- ``z = np.empty_like(x); np.add(x, y, out=z)``: leads to no temporary, it is
  one large pass instead of build+copy.
- ``np.putmask(z, (x < 0.25) & (y > 0.5), x*y + 1.0)``: mask, not scatter;
  avoids index arrays and irregular writes.
- ``z[gt1] += 2.0; z[~gt1] -= 2.0``: leads to two whole-array updates, no
  ``CHUNK = 4096`` loop, which means no thousands of tiny tasks.

Utility
^^^^^^^

.. image:: ../_images/profiling_debugging/utility_efficient.png
   :alt: Efficient utility lane with short bursts
   :width: 90%

Why this is good:
Quiet baseline with brief bursts at the end. Mapping/scheduling is compact;
most time is in real compute. The late burst corresponds to final mapping/sync
before completion. Little “confetti” in the utility lanes means few meta-tasks;
dependencies are simple and batched. Low avg line except at startup/teardown
mean orchestration cost is small vs. compute.

Efficient Code:

- ``np.add(x, y, out=z)``: one big operation; fewer instances to map/track.
- ``np.putmask(z, cond, x*y + 1.0)``: mask, not scatter; avoids index arrays
  and per-slice dependency checks.
- ``z[gt1] += 2.0; z[~gt1] -= 2.0``: two whole-array updates, no
  ``CHUNK = 4096`` loop which means no thousands of tiny tasks.

I/O
^^^

.. image:: ../_images/profiling_debugging/io_efficient.png
   :alt: Efficient I/O lane with minimal top-level overhead
   :width: 90%

Why this is good:
The lane is dominated by a long, tall plateau that is a single TopLevelTask
block with only a few short blips for init/teardown, there is no mid-run I/O
plateaus. The avg line stays flat/low between blips with no steady chatter,
this means host to device copies are not here (they’d appear in Channel, which
stays quiet). Top-level orchestration is minimal, the time goes to compute,
not file I/O or driver overhead.

Efficient Code:

- ``np.add(x, y, out=z)``: writes directly to a preallocated output; avoids
  extra writes/allocs.
- ``np.putmask(z, cond, x*y + 1.0)``: mask, not scatter; no index arrays,
  fewer driver events.
- ``z[gt1] += 2.0; z[~gt1] -= 2.0``: two whole-array updates, no
  ``CHUNK = 4096`` loop, so the runtime doesn't generate many tiny top-level
  actions (Channel also stays free of thin, persistent copy baselines).

System
^^^^^^

Why this is good:
Near-zero for most of the run, with only a gradual rise to ~8% late in the
timeline (allocator growth/instance finalization/teardown). No mid-run
plateaus, the OS/allocator work isn't the bottleneck; compute and bulk copies
dominate.

Efficient Code:

- ``np.add(x, y, out=z)``: avoids an extra temporary/allocation.
- ``np.putmask(z, cond, x*y + 1.0)``: mask update, no large index arrays to
  allocate/manage.
- ``z[gt1] += 2.0; z[~gt1] -= 2.0``: two whole-array updates, no
  ``CHUNK = 4096`` loop, so far fewer small allocation/synchronization points
  at the system.

Channel (chan)
^^^^^^^^^^^^^^

.. image:: ../_images/profiling_debugging/channel_efficient.png
   :alt: Efficient Channel lane with a few bulk transfers
   :width: 90%

Why this is good:
Quiet baseline for most of the run; no thin, persistent copy noise. A couple
of tall plateaus only when needed for bulk transfers/flush at the end.
Indicates high effective throughput: few large DMA copies, minimal per-copy
overhead, and little sync pressure on Utility/CPU.

Efficient Code:

- ``np.add(x, y, out=z)``: writes once into a preallocated output; avoids
  extra traffic.
- ``np.putmask(z, cond, x*y + 1.0)``: boolean mask, not scatter; no irregular
  index copies.
- ``z[gt1] += 2.0; z[~gt1] -= 2.0``: two whole-array updates, no
  ``CHUNK = 4096`` loop, this eliminates floods of tiny copies.


Efficient Multi-GPU Results - (4 Ranks 1 GPU each)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All ranks:

.. image:: ../_images/profiling_debugging/gpu_efficient_all_ranks.png
   :alt: Efficient multi-GPU profiler view across ranks
   :width: 90%

GPU Dev
^^^^^^^

Why this is good:
Steady compute time: The green “avg” line goes high and stays high while work
runs. That means the GPU is busy doing math, not waiting around. Few, wide
kernels: Solid, thick bars mean big kernels that do lots of work per launch
(less start/stop overhead). Gaps between kernels are short, showing good
overlap with transfers and low idle time. Device time is spent on real
computation (vector add, masked overwrite, threshold updates) instead of
launch/sync overhead.

Efficient Code:

- ``np.add(x, y, out=z)``: launches a single wide vector add kernel, not
  build+copy+separate add.
- ``np.putmask(z, cond, x*y + 1.0)``: compiles to one masked overwrite
  kernel; avoids scatter that would fragment into many micro-kernels.
- ``z[gt1] += 2.0; z[~gt1] -= 2.0``: just two whole-array updates, not
  thousands of ``CHUNK = 4096`` slice updates.

GPU Host
^^^^^^^^

Why this is good:
Mostly quiet baseline with a few short bursts aligned to device kernels
showing minimal launch/orchestration overhead. Avg line stays low between
bursts; no comb/“barcode” pattern of micro-launches. The host is mostly idle
while the GPU runs long kernels, which is exactly what you want. Clear
separation of roles: CPU briefly issues work; GPU does the heavy lifting.

Efficient Code:

- ``np.add(x, y, out=z)``: one large launch, not build+copy+extra kernel.
- ``np.putmask(z, cond, x*y + 1.0)``: mask update (no scatter/nonzero),
  avoiding extra setup and multiple small launches.
- ``z[gt1] += 2.0; z[~gt1] -= 2.0``: just two whole-array updates, no
  ``CHUNK = 4096`` loop means orders of magnitude fewer launches.

Framebuffer
^^^^^^^^^^^

Why this is good:
Low flat line that stays at ~0% utilization most the run then gradually builds
up to ~1% utilization at the end, showing memory management isn't a bottleneck.
Alloc/teardown bumps are short; there's no mid-run allocation mess, so data
lives in device memory while kernels run.

Efficient Code:

- ``np.add(x, y, out=z)``: writes into a preallocated output (no extra
  full-size temporary to allocate/free).
- ``np.putmask(z, cond, x*y + 1.0)``: mask update, not scatter (avoids
  partitioned/irregular storage and extra instances).
- ``z[gt1] += 2.0; z[~gt1] -= 2.0``: two whole-array passes, no
  ``CHUNK = 4096`` loop (prevents thousands of small ephemeral instances).

Zerocopy
^^^^^^^^

Why this is good:
Avg stays at ~0% utilization for the entire run, Zerocopy traffic is
negligible. Only a few short alloc/free ticks near the end; no background
chatter. No measurable Zerocopy activity. That means that Zerocopy wasn’t used
for steady data movement. Instead, data was staged in device memory and moved
through the normal Channel (DMA) path, with no measurable reliance on pinned
host memory.

Efficient Code:

- ``np.add(x, y, out=z)``: computes in-place into a preallocated device array,
  avoiding extra host to device touches.
- ``np.putmask(z, cond, x*y + 1.0)``: mask update (no ``nonzero + scatter``),
  preventing irregular host-pinned traffic.
- ``z[gt1] += 2.0; z[~gt1] -= 2.0``: two whole-array kernels (no CHUNK loop),
  so there aren’t many tiny host-access events to begin with.


Profilers - Wrap Up
-------------------

By using Legate’s built-in profiler, you gain the ability to uncover hidden
bottlenecks and inefficiencies in your code. Profiling doesn't just expose
“bugs”, it provides a lens to reason about performance and systematically
improve it. What looks like small structural tweaks (fusing operations,
avoiding scatter writes, and cutting temporaries), translates into fewer
tasks, less orchestration, and higher throughput. This results in a clear
transition between average code that “just runs” to efficient, scalable, and
production-ready code. Profiling turns performance tuning from guesswork into
an intentional, data-driven process that elevates code quality from functional
to excellent.

Inefficient (CPU)                               Efficient (CPU)