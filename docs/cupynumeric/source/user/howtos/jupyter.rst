Configuring Jupyter kernels
===========================

Legate supports single-node execution of programs using Jupyter Notebooks.
Please use the instructions given below to set up IPython kernels that
will be used in the notebooks.

Setup
-----

IPython Kernel
~~~~~~~~~~~~~~

Inputs that are passed to the Legate launcher will now be passed to the
notebook through IPython kernels. By default, ``LEGATE_SM_GPU`` kernel will
be available and set to use one GPU.

For each set of inputs to legate, a new kernel will have to be created using
``legate-jupyter`` and then selected from the drop-down menu for
"Select Kernel" from your notebook.

Use the following to list all the installed kernels. By default,
``LEGATE_SM_GPU`` should be available.

.. code-block:: sh

    jupyter kernelspec list

To create a new kernel that corresponds to a particular set of inputs to
``legate``, say, to run on 2 CPUs with 10GB of memory and 10% of memory
reserved for eager allocations, run the following:

.. code-block:: sh

    legate-jupyter --name "legate_cpus_2" --cpus 2 --sysmem 10000 --eager-alloc-percentage 10

    jupyter kernelspec list

This should create a new kernel named ``legate_cpus_2``. The installed kernel
can then be selected from the notebook to run on two CPUs.

You can also see input arguments that were passed to Legate by the kernel by
using magic commands from a cell in the notebook (including the % character),
like below:

.. code-block:: text

    %load_ext legate.info
    %legate_info

A sample output from a custom kernel is given below:

.. code-block:: text

    Kernel 'legate_cpus_2' configured for 1 node(s)

    Cores:
    CPUs to use per rank : 2
    GPUs to use per rank : 0
    OpenMP groups to use per rank : 0
    Threads per OpenMP group : 4
    Utility processors per rank : 2

    Memory:
    DRAM memory per rank (in MBs) : 10000
    DRAM memory per NUMA domain per rank (in MBs) : 0
    Framebuffer memory per GPU (in MBs) : 4000
    Zero-copy memory per rank (in MBs) : 32
    Registered CPU-side pinned memory per rank (in MBs) : 0

Running on a remote server
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you intend to run the notebook on a remote server or a laptop, you will
have to create a tunnel from your localhost to the remote server. Substitute
remote-server-hostname with the hostname of the remote server you plan to use,

.. code-block:: sh

    ssh -4 -t -L 8888:localhost:8002 username@remote-server-hostname ssh -t -L 8002:localhost:8888 remote-server-hostname

and then run on your local machine:

.. code-block:: sh

    jupyter notebook --port=8888 --no-browser

This should give a URL where the Jupyter server is running and will look like
this:

.. code-block:: text

    http://localhost:8888/tree?token=<token-id>

Where ``<token-id>`` will be different each time you launch jupyter. Launch
the URL from your browser and choose the ``Legate_SM_GPU`` kernel. This ensures
that the underlying computations can be run using the resources specified
in the ``Legate_SM_GPU`` kernel.

For more information on how this works with the runtime, we refer the readers
to respective sections in Legion and Legate documentation.

Running Jupyter Notebooks
-------------------------

You are now set up to run the notebooks using Jupyter with your configured
options. Check out the notebooks in the `examples` section.
