Running tests
=============

Basic usage
-----------

The simplest way to run the cuPyNumeric test suite is to use the ``test.py``
test driver script.

.. code-block:: sh

   ./test.py --color --debug

This will start test execution for the default CPU stage, as well as enabling
additional debug output:

.. code-block:: sh

    ################################################################################
    ###
    ### Test Suite Configuration
    ###
    ### * Feature stages       : cpus
    ### * Test files per stage : 135
    ### * TestSystem description   : 6 cpus / 2 gpus
    ###
    ################################################################################

    ################################################################################
    ### Entering stage: CPU (with 2 workers)
    ################################################################################

    [PASS] (CPU) examples/benchmark.py
    [PASS] (CPU) examples/black_scholes.py
    [PASS] (CPU) examples/cg.py
    [PASS] (CPU) examples/cholesky.py
    [PASS] (CPU) examples/einsum.py

After all the tests have finished a stage summary will display, including details of
any test failures, as well as an overall summary for all stages combined:

.. code-block:: sh

    [PASS] (CPU) tests/integration/test_view.py
    [PASS] (CPU) tests/integration/test_vstack.py
    [PASS] (CPU) tests/integration/test_where.py
    [PASS] (CPU) tests/integration/test_window.py
                                    CPU: Passed 135 of 135 tests (100.0%) in 196.08s

    ################################################################################
    ###
    ### Exiting stage: CPU
    ###
    ### * Results      : 135 / 135 files passed (100.0%)
    ### * Elapsed time : 0:03:16.075726
    ###
    ################################################################################

        ----------------------------------------------------------------------------

    ################################################################################
    ###
    ### Overall summary
    ###
    ### * CPU   : 135 / 135 passed in 196.08s
    ###
    ### All tests: Passed 135 of 135 tests (100.0%) in 196.08s
    ###
    ################################################################################

Stage configuration
-------------------

There are four possible test stages that can be run:

* ``cpus`` runs the test stage using CPUs

* ``cuda`` runs the test stage using GPUs

* ``openmp`` runs the test stage using OpenMP

* ``eager`` runs the test stage using plain Numpy

The stages are specified via the ``--use`` command line parameter, and may be
combined in a comma separated list to run multiple stages, for example:

.. code-block:: sh

    legate test.py --use=cpus,eager,cuda,openmp

There are a number of options to control the runtime configuration for
different stages:

--cpus CPUS
  Number of CPUs per node to use

--gpus GPUS
  Number of GPUs per node to use

--fbmem FBMEM
  GPU framebuffer memory (MB)

--omps OMPS
  Number of OpenMP processors per node to use

--ompthreads THREADS
  Number of threads per OpenMP processor

--utility UTILITY
  Number of cores to reserve for runtime services

There are also options to get more verbose or color-coded terminal output:

--color
  Whether to use color terminal output (if colorama is installed)

-v, --verbose
  Display verbose output. Use -vv for even more output (test stdout)

--dry-run
  Print the test plan but don't run anything

--debug
  Print out the commands that are to be executed


for full details see the output of ``test.py --help``.