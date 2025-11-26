Understanding and Handling Out-of-Memory (OOM) Issues – Example 2
------------------------------------------------------------------

How OOM Occurs
~~~~~~~~~~~~~~

cuPyNumeric runs on top of Legate Core. At launch, the ``legate`` launcher
auto-sizes memory pools for each “memory kind” it detects (e.g., CPU
``SYSTEM_MEM``, and GPU framebuffer) on the assigned process/GPU. You can
override these defaults to fixed sizes if needed with flags such as
``--sysmem`` (MiB of host DRAM) and ``--fbmem`` (MiB of GPU memory). If an
operation needs to create a new instance that exceeds the reserved capacity of
a pool, the runtime raises an out-of-memory error for that memory kind (e.g.,
``SYSTEM_MEM`` or ``FBMEM``) and reports the task/store that triggered it.

Why this matters: Most “mystery OOMs” aren’t total node exhaustion, they’re
per-process, per-kind pool exhaustion. The fix is often to:

1. Right-size those pools.
2. Reduce peak live instances so they fit.

For more information on Legate Core visit: Overview — NVIDIA legate.core


Demo Script
~~~~~~~~~~~

We’ll intentionally run with a tiny ``SYSTEM_MEM`` pool to trigger a controlled
OOM.

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

``LEGATE_TEST=1`` enables diagnostic/verbose mode: detailed allocation
information such as logical store creation, instance sizes, and memory
reservations, as opposed to a brief undescriptive error message.

``legate --cpus 1 --gpus 0 --sysmem 40 oom.py`` runs the script ``oom.py``
with one CPU worker and a fixed system memory pool of 40 MiB. Legate will
pre-allocate a 40 MiB region from host DRAM to use for all CPU-side array
instances (``SYSTEM_MEM``). Any time an operation requires more than this
reserved pool, you’ll see a ``Failed to allocate… of kind SYSTEM_MEM`` error.

GPU run (FBMEM behavior)
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Single GPU, intentionally tight framebuffer pool
   LEGATE_TEST=1 legate --cpus 2 --gpus 1 --fbmem 40 --sysmem 512 oom.py

Tip: Flags are per process. If you use multiple ranks per node, each rank needs
its own slice of ``--sysmem`` / ``--fbmem``.

Steps to Diagnose OOM
---------------------

Step 1. Read The OOM Line
~~~~~~~~~~~~~~~~~~~~~~~~~

When an OOM happens, the failure line will tell you the memory kind that ran
out (e.g., of kind ``SYSTEM_MEM`` or GPU framebuffer) and which task/logical
store was being created when it failed. That points you at the operation that
spiked usage.

OOM Error Message (CPU example):

.. code-block:: text

   Failed to allocate 8388608 bytes on memory 1e00000000000000 (of kind SYSTEM_MEM) for region requirement(s) {1} of Task cupynumeric::BinaryOpTask[/home/USER/d/cupynumeric/oom.py:16] (UID 8)
   corresponding to a LogicalStore allocated at /home/USER/d/cupynumeric/oom.py:16 There is not enough space because Legate is reserving 33554400 of the available 41943040 bytes for the following LogicalStores:
   LogicalStore allocated at /home/USER/d/cupynumeric/oom.py:13:
     Instance 4000000000000003 of size 8388608 covering elements <0>..<1048575>
         created for an operation launched at /home/USER/d/cupynumeric/oom.py:13
   LogicalStore allocated at /home/USER/d/cupynumeric/oom.py:7:
     Instance 4000000000000002 of size 8388608 covering elements <0>..<1048575>
         created for an operation launched at /home/USER/d/cupynumeric/oom.py:13
     Instance 4000000000000001 of size 8388592 covering elements <1>..<1048574>
         created for an operation launched at /home/USER/d/cupynumeric/oom.py:10
   LogicalStore allocated at /home/USER/d/cupynumeric/oom.py:4:
     Instance 4000000000000000 of size 8388592 covering elements <0>..<1048573>
         created for an operation launched at /home/USER/d/cupynumeric/oom.py:10

Decode Error Message:
^^^^^^^^^^^^^^^^^^^^^

Important failure line:

.. code-block:: text

   Failed to allocate 8388608 bytes on memory 1e00000000000000 (of kind SYSTEM_MEM) for region requirement(s) {1} of Task cupynumeric::BinaryOpTask[/home/USER/d/cupynumeric/oom.py:16] (UID 8)

Interpretation: Legate attempted to allocate an 8 MiB array in the 40 MiB
``SYSTEM_MEM`` pool for the ``BinaryOpTask`` at line 16, but no contiguous
free block was available. The OOM originates from that task.

Segment
"""""""

- Failed to allocate 8388608 bytes

  The runtime tried allocating ~8 MiB for a new array instance. This is the
  size of the region (number of elements * element size).

- on memory 1e00000000000000 (of kind SYSTEM_MEM)

  Internal ID of the memory pool; every memory kind (``SYSTEM_MEM``, ``FBMEM``,
  ``ZCMEM``) has a unique 64-bit handle. ``(of kind ..)`` tells you which
  memory pool failed, here, system memory (CPU DRAM). If it said ``FBMEM``, it
  would be GPU framebuffer memory.

- for region requirement(s) {1}

  Internal bookkeeping number identifying which logical region of the task
  requested the allocation.

- of Task cupynumeric::BinaryOpTask[/home/USER/d/cupynumeric/oom.py:16] (UID 8)

  The task name that triggered the allocation. ``BinaryOpTask`` corresponds to
  a basic elementwise operation in cuPyNumeric (e.g., addition, subtraction).
  ``[/home/USER/…:16]`` is the exact source line that triggered the failed
  operation (``d = c + 3``) in the demo (e.g., ``{"file":
  "/home/USER/d/cupynumeric/oom.py", "line": 16}``). ``UID`` is a unique ID
  assigned to this particular task invocation by the runtime, it is useful
  when correlating with profiler traces.

Rest of OOM Error Message
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   corresponding to a LogicalStore allocated at /home/USER/d/cupynumeric/oom.py:16 There is not enough space because Legate is reserving 33554400 of the available 41943040 bytes for the following LogicalStores:
   LogicalStore allocated at /home/USER/d/cupynumeric/oom.py:13:
     Instance 4000000000000003 of size 8388608 covering elements <0>..<1048575>
         created for an operation launched at /home/USER/d/cupynumeric/oom.py:13
   LogicalStore allocated at /home/USER/d/cupynumeric/oom.py:7:
     Instance 4000000000000002 of size 8388608 covering elements <0>..<1048575>
         created for an operation launched at /home/USER/d/cupynumeric/oom.py:13
     Instance 4000000000000001 of size 8388592 covering elements <1>..<1048574>
         created for an operation launched at /home/USER/d/cupynumeric/oom.py:10
   LogicalStore allocated at /home/USER/d/cupynumeric/oom.py:4:
     Instance 4000000000000000 of size 8388592 covering elements <0>..<1048573>
         created for an operation launched at /home/USER/d/cupynumeric/oom.py:10

Rest of OOM Error Message – Meaning
"""""""""""""""""""""""""""""""""""

- corresponding to a LogicalStore allocated at /home/USER/d/cupynumeric/oom.py:16

  A ``LogicalStore`` is Legate’s internal representation of an array region
  (or “view”) that lives somewhere in memory. This line confirms that the
  store associated with ``oom.py:16`` (``d = c + 3``) is the one that failed.
  The runtime attempted to map that store to physical memory but couldn't
  satisfy the allocation.

  Note: You can now pinpoint the failure to a specific variable (the result of
  ``d = c + 3``) and know it wasn’t an earlier array but a new instance being
  materialized.

- There is not enough space because Legate is reserving 33554400 of the available 41943040 bytes for the following LogicalStores:

  ``41943040`` bytes - Total reserved pool size for ``SYSTEM_MEM`` → 40 MiB
  (``--sysmem 40``).

  ``33554400`` bytes - Amount already reserved/consumed by existing instances
  (about 32 MiB/Mebibyte).

  Note: Out of the 40 MiB pool, roughly 32 MiB is occupied by other arrays.
  The remaining ~8 MiB isn’t a free, contiguous block large enough to hold a
  new instance once alignment and headers are included, and the mapper keeps
  currently mapped instances reserved (non-evictable) while creating the next
  one. See Overview below for more information.

- LogicalStore allocated at /home/USER/d/cupynumeric/oom.py:13:

  .. code-block:: text

     Instance 4000000000000003 of size 8388608 covering elements <0>..<1048575>
         created for an operation launched at /home/USER/d/cupynumeric/oom.py:13

  From this line, details where the allocated memories go.
  ``oom.py:13 = c = b + 2``

  - ``Instance 4000000000000003`` - Internal instance ID. Used internally for
    tracking physical allocations.
  - ``size 8388608`` - 8 MiB allocated
  - ``covering elements <0>..<1048575>`` - Range of local elements this
    instance covers, 1 million elements (0 → 1,048,575).
  - ``created for an operation launched at…`` - Confirms which operation
    produced this instance (line 13).

- LogicalStore allocated at /home/USER/d/cupynumeric/oom.py:7:

  .. code-block:: text

     Instance 4000000000000002 of size 8388608  covering elements <0>..<1048575>
         created for an operation launched at /home/USER/d/cupynumeric/oom.py:13
     Instance 4000000000000001 of size 8388592 covering elements <1>..<1048574>
         created for an operation launched at /home/USER/d/cupynumeric/oom.py:10

  Line 7 corresponds to ``b = np.zeros(...)``.

  Because ``b`` was sliced (``b[1:-1] = a``) and reused (``c = b + 2``),
  multiple physical instances exist for the same logical store ``b``. Each
  instance (~8MiB) represents a materialized subregion or copy created by
  different downstream operations. ``/home/USER/d/cupynumeric/oom.py:10 →
  b[1:-1] = a`` This slice assignment materializes instances for both ``a``
  and the sliced view of ``b``. That's why you see instances tied to line 10
  for the stores at lines 4 (``a``) and 7 (``b``) in the OOM list. Those
  instances stay reserved while later ops run, which is what tightens the
  pool and makes the ``d = c + 3`` allocation at line 16 fail.

- LogicalStore allocated at /home/USER/d/cupynumeric/oom.py:4:

  .. code-block:: text

     Instance 4000000000000000 of size 8388592 covering elements <0>..<1048573>
         created for an operation launched at /home/USER/d/cupynumeric/oom.py:10

  Line 4 is ``a = np.ones(...)``.
  ``a`` remains in memory as an 8 MB instance used earlier by slice assignments.

  Note: ``a``’s memory allocation still exists in the runtime even though it's
  not directly used later, it hasn't been freed because it's referenced by
  ``b[1:-1] = a``.

Overview
^^^^^^^^

The pool reaches capacity because older arrays (``a``, ``b``, ``c``) memory
allocations still exist in the runtime and haven't been released or reclaimed
yet. The new result for ``d = c + 3`` can’t fit at the moment. The “why” is a
mix of pool size and live instances the program keeps around. From the above
descriptions, we can see that previous arrays and allocations take up ~32MiB
out of the reserved pool of 40MiB.

- Pool size (``SYSTEM_MEM``): 40 MiB = 41,943,040 bytes
- Already Reserved: 33,554,400 bytes (~32MiB) across four ~8 MiB instances
  (``a``, ``b`` slice/full, ``c``).
- Leftover: 41,943,040 - 33,554,400 = 8,388,640 bytes

The new instance wants 8,388,608 bytes. That looks like it should fit (32 extra
bytes), but it still fails because the runtime’s alignment and per-instance
bookkeeping make the actual footprint a bit larger than the printed payload
(8,388,608 bytes). So 32 MiB used + 8 MiB new in a 40 MiB pool can still OOM.
A real instance needs payload + per-instance overhead (e.g., internal instance
header/descriptor and alignment padding managed by Realm/Legion). Even a
modest header (> 64–256 bytes, typical for a descriptor + aligned field
layout) pushes the actual requirement to > 8,388,672 bytes, which exceeds the
8,388,640 bytes free. The “reserved … for the following LogicalStores” list
shows the requested instance sizes (the array field bytes). It doesn’t itemize
allocator extras like per-instance headers, layout descriptors, or alignment
padding the Realm/Legion allocator needs to place the instance in that memory.

Step 2. Verify Resource Reservations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Confirm the runtime actually reserved enough memory for your process(es): use
``--show-config``, and remember that flags are per process. When you run
multiple ranks per node, each process needs its own slice of CPU/GPU memory,
sometimes you may even need to reduce per-rank ``--sysmem``/bind CPUs.
``--show-config`` is a fast sanity check that explains an OOM is due to
mis-sizing pools per rank.

- Catches misconfig: Confirms your per-rank
  ``--sysmem`` / ``--fbmem`` / ``--zcmem`` are what you think they are.
  If too big for the node or for R ranks, you’ll OOM regardless of code.
- Disambiguates cause: Distinguishes “pool too small” vs
  “duplicate instances/overlap”. If pools are clearly undersized, resize
  first; else, consider prefetching/other mitigation techniques.
- Clarity: Gives a one-line snapshot to paste into bug reports: exact pool
  sizes by memory & rank.

.. code-block:: bash

   # legate --show-config
   # print the pools you'd use.. "&&" ..then run the repro with verbose OOM info:
   legate --cpus 1 --gpus 0 --sysmem 40 --show-config \
   && LEGATE_TEST=1 legate --cpus 1 --gpus 0 --sysmem 40 oom.py

Confirm the per-kind pool sizes match your flags and that each rank has
sensible values. (If using ``-n`` or multiple ``--ranks-per-node``, scale your
expectations.) An example would look something like:

.. image:: ../_images/profiling_debugging/show_config.png
   :alt: Example legate --show-config output
   :width: 90%

CPU Run:

CUDA_ERROR_NO_DEVICE: Harmless in this context, we asked for
``--gpus=0`` so Legion/Realm reports “no device” and proceeds on CPU only.
Same for “..not able to discover the CUDA resources”.

Step 3. Sanity-Check Device Memory Externally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On GPU nodes, also glance at ``nvidia-smi``, or:

.. code-block:: bash

   cat /proc/meminfo | grep MemAvailable

to confirm there’s headroom/memory on each selected device. If you OOM while
directly allocating memory, check if there's headroom to increase memory
allocation for your run. See visual examples below:

CPU (host RAM):

.. code-block:: bash

   cat /proc/meminfo | grep MemAvailable

.. image:: ../_images/profiling_debugging/meminfo.png
   :alt: Example MemAvailable output
   :width: 70%

GPU (device VRAM):

.. code-block:: bash

   nvidia-smi

.. image:: ../_images/profiling_debugging/nvidia_smi.png
   :alt: Example nvidia-smi output
   :width: 70%

- Per-GPU Memory-Usage (1 MiB / 40960 MiB): shows headroom.
- Per-GPU rows (0–3, A100-SXM4-40GB): Model and count.
- Processes: No running processes found, confirms nothing else is using the
  GPUs.

Mitigation Strategies
---------------------

Depending on the root cause you analyzed from the OOM message or other
diagnostic technique, there are different mitigations you can take. Parts A–D
below will walk you through which mitigation technique to use and when. Note
that these mitigation strategies are not mutually exclusive, most workloads
benefit from a combination of mitigations rather than a single one.

A. Resize Legate’s Memory Reservations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, Legate uses all available per-rank memory kinds (``SYSTEM_MEM``,
``FBMEM``, ``ZCMEM``) unless you constrain them. In some cases, memory is
already used by other processes, so Legate cannot reserve as much as it wants
and you see an OOM. In this case, use ``--sysmem`` / ``--fbmem`` (and
optionally ``--zcmem``) to size pools explicitly.

(i) When to increase memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use larger pools when the per-rank working set simply needs more space and the
node/device has headroom. This would be a case where you are explicitly
allocating/constraining memory.

- OOM cites “of kind SYSTEM_MEM / FBMEM”, and nvidia-smi/MemAvailable show
  free memory.
- You keep multiple large arrays/live instances by design (e.g., prefetching
  whole arrays, big intermediates).
- You already minimized temporaries/scatter/tiny tasks, but still run into
  pool limits.

How to increase per-rank reservations: ``--sysmem <MiB>`` (host DRAM),
``--fbmem <MiB>`` (GPU VRAM), optionally ``--zcmem <MiB>``.

(ii) When to decrease memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Shrink pools to fit scheduler limits and leave headroom for other processes,
or to encourage spill to host. This would be a case where Legate is
automatically sizing memory pools.

- Reservation fails at startup (can’t pre-reserve), or you’re on shared/MIG
  GPUs with tighter per-process caps.
- Many ranks per node: ``R × --fbmem`` / ``R × --sysmem`` would exceed
  device/host capacity.
- You want less VRAM pinned (e.g., prefer host placement/offload) or zero-copy
  was oversized for the workload.

How to decrease per-rank reservations or ranks: (lower)
``--fbmem <MiB>``, ``--sysmem <MiB>``, and/or ``--zcmem <MiB>``; or reduce
``--ranks-per-node``.

Per-rank rule: Pools are per process, If you’re launching multiple processes
per node, reduce per-rank reservations or the number of ranks
(``--ranks-per-node``). Your per-rank ``--fbmem`` must fit under what the
scheduler can give each rank/device.

B. Prefetch The Data
~~~~~~~~~~~~~~~~~~~~

Prefetching is an optimization technique that involves fetching data and
loading it into memory before it is requested. By proactively materializing
the data to the target memory, along with the required slice ranges, before
the heavy compute, the runtime avoids creating duplicate physical instances of
the same logical array mid-compute, preventing peak-memory spikes and OOM.

When to use Prefetching:
^^^^^^^^^^^^^^^^^^^^^^^^

(Example from previous 40 MiB ``SYSTEM_MEM`` run)

.. code-block:: text

   Legate is reserving 33554400 of the available 41943040 bytes for LogicalStores:
   LogicalStore allocated at /home/USER/d/cupynumeric/oom.py:13:
     Instance 4000000000000003 of size 8388608 covering elements <0>..<1048575>
         created for an operation launched at /home/USER/d/cupynumeric/oom.py:13
   LogicalStore allocated at /home/USER/d/cupynumeric/oom.py:7:
     Instance 4000000000000002 of size 8388608 covering elements <0>..<1048575>
         created for an operation launched at /home/USER/d/cupynumeric/oom.py:13
     Instance 4000000000000001 of size 8388592 covering elements <1>..<1048574>
         created for an operation launched at /home/USER/d/cupynumeric/oom.py:10

Refer to the OOM error log: if the OOM log shows the same ``LogicalStore``
being instantiated for overlapping/expanding ranges. For example, the slice
assignment at line 10 creates a ``b`` instance covering ``<1>..<1048574>``;
later, ``c = b + 2`` at line 13 forces a full-range ``b`` instance
``<0>..<1048575>`` under the ``b`` store that originated at line 7 (created by
the op at line 13), and a separate full-range result instance for ``c`` (the
store that originates at line 13). This means the runtime has to allocate a
bigger instance while a smaller one is still live → temporary duplication →
peak spike → OOM. In the profiler, you’ll see transfers/instance creation
appear inside the thick compute band (bad). After prefetch, they should occur
before the band; the channel/transfer lanes are quiet during kernels.

Technique 1 - cuPyNumeric ``stencil_hint`` (prefetch for stencil/halo ranges)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

API (cuPyNumeric ndarray method):

.. code-block:: python

   def stencil_hint(
           self, low_offsets: tuple[int, ...],
           high_offsets: tuple[int, ...]) -> None:

       """
       Inform cuPyNumeric that this array will be used in a stencil computation.

       This allocates space for “ghost” elements ahead of time, rather than
       discovering the full extent incrementally, avoiding intermediate copies.
       """

What it does:
"""""""""""""

Declares a halo (ghost cells), which is a thin border of extra elements around
each partition that holds copies of neighboring data your stencil will read
(e.g., left/right or N/S/E/W). By specifying ``low_offsets``/``high_offsets``,
cuPyNumeric materializes one larger backing instance up front that already
includes this halo, rather than discovering the full extent incrementally.
This avoids intermediate copies and mid-compute growth (small instances
growing to large instances), reducing peak memory spikes and helping prevent
OOMs.

Parameters:
"""""""""""

- ``low_offsets``: per-dimension halo toward the negative direction. Negative
  direction refers toward smaller indices on that axis.
- ``high_offsets``: per-dimension halo toward the positive direction. Positive
  direction refers toward larger indices on that axis.

Examples:

- 1D: ``low_offsets=(1,), high_offsets=(2,)`` → pre-allocate room for neighbors
  ``i-1`` (one to the left) and ``i+1``, ``i+2`` (two to the right).
- 2D (shape ``[rows, cols]``): ``low_offsets=(1, 2), high_offsets=(3, 1)`` →
  Add halo up 1 row and left 2 cols (negative), and down 3 rows and right 1
  col (positive).

Note: Call ``stencil_hint`` before the stencil section that uses
overlapping/expanding slices. Be slightly conservative: if you might touch up
to 2 cells in a direction, pass 2. Current limitation: behavior may not match
expectations when multiple CPU/OpenMP processors share the same memory.

Example: 1D
"""""""""""

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

Why this works: Without the hint, the runtime may first create a smaller
instance for a subrange (e.g., ``b[1:-1]``) and later a larger one (full
``b``) while the smaller is still live, temporarily doubling the footprint.
``stencil_hint`` allocates the larger instance once before compute, so
downstream ops reuse it and no mid-band growth occurs.

Note: This technique, when run in place of our original example with the same
memory allocation ``LEGATE_TEST=1 legate --cpus 1 --gpus 0 --sysmem 40
oom.py``, easily passes without an OOM.

Technique 2 - cuPyNumeric Prefetch via a whole-array touch (no temporaries)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you decide not to call the cuPyNumeric API or you are unsure which slices
will be touched, stage the entire array once with a no-op ufunc that touches
every element without allocating a new array. This is considered prefetching
because you are materializing before compute:

.. code-block:: python

   import cupynumeric as np

   N = 1024 * 1024
   a = np.ones((N - 2,), dtype=np.float32)
   b = np.zeros((N,), dtype=np.float32)

   # GPU: materializes in FB_MEM; CPU-only: materializes in SYSTEM_MEM
   np.multiply(b, 1, out=b)      # or: np.add(b, 0, out=b)   # or: b *= 1

   b[1:-1] = a
   c = b + 2
   d = c + 3

Why this works: Using ``out=`` (or in-place) guarantees no second full-size
temporary is created while you prefetch. For NumPy-style ufuncs in
cuPyNumeric (e.g., ``add``, ``multiply``), passing ``out=arr`` tells the
runtime to write results directly into ``arr``’s existing buffer. No new
``n``-element result array is allocated; the kernel reads and writes in place.
The “prefetch” touch is a no-op math pass that forces materialization on the
target memory. With ``out=``, that pass reuses the same storage, so you get
the placement effect without creating a second full-size array. After this,
run your heavy ops; the instance already exists at the needed size, so there's
no mid-band growth.

Note: This technique, when run in place of our original example with the same
memory allocation ``LEGATE_TEST=1 legate --cpus 1 --gpus 0 --sysmem 40
oom.py``, easily passes without an OOM.

C. Releasing Memory Between Phases (del, GC, and allocator pools)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dropping references, collecting garbage, and flushing allocator caches shrinks
the live working set so the next phase has the headroom and is less likely to
hit OOM or suffer from cache-induced slowdowns.

1) Drop references:
"""""""""""""""""""

The ``del`` statement in Python deletes a reference to an object. It removes
the binding between a variable name and the object it refers to in the current
namespace. It will only delete the object if there are no other references to
it. ``del`` does not “free memory” by itself; it just removes a single
reference. An object is actually freed once no references remain. In CPython
(reference implementation of Python) that usually happens immediately via
reference counting; if there are reference cycles, the garbage collector (GC)
may be needed.

- Delete all names that point to large objects.
- After ``del``, the object may still exist if another variable/container
  references it.

2) Run the garbage collector:
"""""""""""""""""""""""""""""

Some objects participate in reference cycles and won't be reclaimed by
refcounts (reference counting) alone. Calling ``gc.collect()`` forces a cycle
detection pass and frees anything that's unreachable. This can reduce your
live Python heap between phases and reclaim memory by cleaning up objects
that are no longer in use.

3) Flush allocator pools (if also using CuPy):
""""""""""""""""""""""""""""""""""""""""""""""

If your process uses CuPy arrays or kernels alongside cuPyNumeric, CuPy’s
device/pinned memory pools may hold on to large caches.

- ``MemoryPool.free_all_blocks()`` releases cached device allocations back to
  the CUDA driver.
- ``PinnedMemoryPool.free_all_blocks()`` releases cached pinned host buffers.

Note: This frees library caches, not your Python objects, and it doesn't
change Legate’s reserved pool sizes (``--sysmem`` / ``--fbmem`` /
``--zcmem``). It just makes more room inside those pools for the next phase.

.. code-block:: python

   # 1) Drop references
   big = None                # break the reference
   if 'big' in globals(): del big
   cache.clear()             # if you stored big arrays in dicts/lists/closures

   # 2) Reclaim cyclic garbage
   import gc
   gc.collect()

   # 3) If you used CuPy in this process, flush its pools
   try:
       import cupy as cp
       cp.get_default_memory_pool().free_all_blocks()
       cp.get_default_pinned_memory_pool().free_all_blocks()
   except Exception:
       pass  # CuPy not used/installed, or no pools to flush

D. Offload to CPU Memory
~~~~~~~~~~~~~~~~~~~~~~~~

If you went through mitigation strategies A–C, all memory is clean, you still
need some data to be in memory, “Offloading” is a way to release some GPU
memory. “Offloading to CPU” means the runtime migrates the contents of an
array from GPU device memory to host memory (RAM). The data will be
automatically moved back to the GPU later, if necessary for an operation.

Offloading with the Legate ``offload_to`` API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In your Legate build (cuPyNumeric 25.11+), you can use the helper
``offload_to`` from ``legate.core.data_interface``:

.. code-block:: python

   import cupynumeric as np
   from legate.core import StoreTarget
   from legate.core.data_interface import offload_to

   N = 1024 * 1024
   a = np.ones((N - 2,), dtype=np.float32)
   b = np.zeros((N,), dtype=np.float32)

   # Work that builds pressure in FBMEM (on GPU runs)
   b[1:-1] = a
   c = b + 2

   # Offload 'c' (or any large array you won't need on GPU immediately) to host RAM:
   offload_to(c, target=StoreTarget.SYSMEM)   # evicts any GPU copies and keeps only a host copy

   # Continue your pipeline; GPU copies will be re-created only if/when needed
   d = c + 3

What this does:
"""""""""""""""

``offload_to`` copies an array to target memory (e.g., system RAM) and
discards any other copies the runtime holds (e.g., in GPU framebuffer). That
immediately frees VRAM for later GPU work. ``StoreTarget.SYSMEM`` targets CPU
RAM. Other options include ``FBMEM`` (GPU VRAM) and ``ZCMEM`` (pinned host
memory for zero-copy). The call makes the CPU copy exclusive (VRAM copies are
discarded), which is what frees space.

Important: the runtime doesn't pre-check capacity. If the target memory lacks
space, your program can still fail. Make sure the ``--sysmem`` is large enough
before offloading.

Trade-off: spilling over to host can save you from OOM but may cost performance
if frequent transfers are needed.

E. If Applicable: Coding Practices to Reduce Peak Memory (Indirect OOM handling)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If mitigation strategies for section A–D aren't enough and you have access to
the code, you can often avoid OOM by lowering the peak working set (the sum of
all live instances at once). Some examples may include (See Example 1 for more
context):

- Avoiding large index arrays and scatter writes; prefer boolean mask over
  ``nonzero(...)`` + advanced indexing.
- Avoiding per-slice “tiny task” loops on big arrays; use whole-array
  vectorized operations instead.
- Avoiding unnecessary temporaries; write into preallocated outputs (via
  ``out=``) instead.

The Legate Profiler can indirectly assist with an OOM diagnosis by finding
inefficiencies in the code. The crash line names the kind of memory that fails
(``SYSTEM_MEM``/``FBMEM``); the profiler will show the time window just before
the crash where Framebuffer/System utilization ramps and stays high while
Channel/Utility stays active. This points to extra temporaries,
scatter/advanced indexing, per-chunk loops, or host to device back and forth,
that keep too many instances live at once and push the pool over capacity.
Launch profiler with ``--profile``, & view the ``legate_*.prof`` files:

- Pinpoint the phase that grows memory in the timeline, a rising plateau in
  Framebuffer (GPU) or System (host) lanes right before failure marks the
  phase that inflated memory.
- Channel (DMA copies): a thin, persistent baseline means constant back and
  forth between host and device (often a cause of scatter patterns/advanced
  indexing), which can mean tons of small transfers which force more data to
  be live at the same time and in more places (host & device duplicates) which
  can cause OOM errors.
- Utility: a “confetti” of meta tasks usually correlates with lots of tiny
  operations (per-slice loops) that materialize extra temporaries/instances.

Last resort: Downcast to a smaller data type to cut memory usage in half
(e.g., ``float32 ← float64`` / ``float16 ← float32``) when numerically
acceptable. Understand by doing so you reduce overall accuracy and dynamic
range, and some operations may upcast internally or lose stability. This is
normally not recommended as most precisions are set for a reason. Prefer mixed
precision (keep accumulators/reductions in ``float32``) if full downcasting is
too risky.

Refer to Example 1 – Profiling cuPyNumeric Applications with Legate Profilers
for this section to view examples & visualizations.


A Few Examples For Applying Different Mitigations
-------------------------------------------------

From our earlier example:

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

What was observed (Steps to Diagnose OOM):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Step 1 (Read the OOM line): The failure is pool exhaustion, instances from
  ``a``, ``b``, ``c`` are still live, so ``d = c + 3`` can't fit in the
  reserved pool at that moment.
- Step 2 (Verify Resource Reservations): Pool sizes per rank match flags.
- Step 3 (Sanity-check headroom): Host/GPU still have capacity, so the issue
  is the per-process pool and peak live instances, not total node memory.

Fix Path A: Resize per-rank pools (Mitigation A: Resize Legate’s Memory Reservations)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Easy and Quick: Give the mapper more headroom in the memory kind that failed,
so the next instance places successfully.

CPU-only (increase ``SYSTEM_MEM``):

.. code-block:: bash

   LEGATE_TEST=1 legate --gpus 0 --cpus 1 --sysmem 128 oom.py

Single-GPU (tight but sane pools; allow host spill):

.. code-block:: bash

   LEGATE_TEST=1 legate --gpus 1 --cpus 2 --fbmem 128 --sysmem 512 oom.py

Flags are per rank; if you run R ranks per node, ensure ``R × --sysmem`` and
``R × --fbmem`` fit real host/GPU capacity (e.g., 2 ranks = 2 × 128 fbmem =
256 fbmem).

Fix Path B: Prefetch Using Technique 2 - Whole Array Touch (Mitigation B: Prefetch the Data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When diagnosing an OOM, and you come across duplicated instances, consider
prefetching. By proactively materializing the data to the target memory, along
with the required slice ranges prior to heavy computation, the runtime avoids
creating duplicated physical instances of the same logical array mid-compute.
This prevents peak-memory spikes and OOM.

.. code-block:: python

   import cupynumeric as np

   N = 1024 * 1024
   a = np.ones((N - 2,), dtype=np.float32)
   b = np.zeros((N,), dtype=np.float32)

   # GPU: materializes in FB_MEM; CPU-only: materializes in SYSTEM_MEM
   np.multiply(b, 1, out=b)      # or: np.add(b, 0, out=b)   # or: b *= 1

   b[1:-1] = a
   c = b + 2
   d = c + 3

Note: In practice, using ``stencil_hint`` is definitely the preferred and more
principled prefetching strategy. The whole-array touch shown here works for
this simple example, but stencil-based prefetching is generally safer and more
scalable for real workloads where slice ranges and halo regions matter.

Fix Path C: Reduce the live working set (Mitigation C: Releasing Memory Between Phases (del, GC, and allocator pools))
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shrink peak live instances so the same pool size suffices, no new memory
required. Replace temporaries with in-place ops and drop unneeded references:

.. code-block:: python

   import cupynumeric as np, gc

   a = np.ones((1024 * 1024 - 2,))
   b = np.zeros((1024 * 1024,))

   # stage, then drop 'a' to reduce the live set (C.1 + C.2)
   b[1:-1] = a
   a = None
   gc.collect()

   # avoid creating 'c' and 'd' instances: in-place updates
   np.add(b, 2, out=b)   # replaces: c = b + 2
   np.add(b, 3, out=b)   # replaces: d = c + 3

   # optional: if CuPy is also in this process, free its caches (C.3)
   try:
       import cupy as cp
       cp.get_default_memory_pool().free_all_blocks()
       cp.get_default_pinned_memory_pool().free_all_blocks()
   except Exception:
       pass

Quick re-checks before rerun (Steps to Diagnose OOM: Steps 2–3):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   legate --show-config                  # pools per rank match intent?
   cat /proc/meminfo | grep MemAvailable # host headroom
   nvidia-smi                            # GPU headroom

Result: Path A expands the pool; Path B & C lowers peak usage. Either resolves
the example OOM for ``d = c + 3`` and makes the cause and fix explicit.

Important: These mitigation strategies are being implemented on a very simple
example. For more complex, larger programs, consider also using offloading, on
top of or instead of some of these techniques.
