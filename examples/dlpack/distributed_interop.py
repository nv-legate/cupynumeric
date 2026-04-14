# Copyright 2026 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Distributed interop between Legate and PyTorch.
#
# Each process runs its own Legate runtime and participates in a
# torch.distributed process group. Data flows between the two runtimes
# via DLPack within each process. Cross-rank communication uses
# torch.distributed collectives.
#
# Two driver modes (--driver legate | --driver pytorch):
#
# Run (legate driver):
#   legate --nodes 2 --ranks-per-node 4 --cpus 1 --gpus 1 \
#       --fbmem 1024 --sysmem 512 --min-gpu-chunk 1 --gpu-bind 0/1/2/3 \
#       --launcher srun examples/dlpack/distributed_interop.py --driver legate
#
# Run (pytorch driver):
#   srun --ntasks-per-node=1 torchrun \
#       --nproc_per_node=4 --nnodes=2 \
#       --node_rank=$SLURM_NODEID \
#       --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#       examples/dlpack/distributed_interop.py --driver pytorch

import argparse
import os
import socket

# Parse driver mode and network config early, before any legate/cupynumeric imports.
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--driver", choices=["legate", "pytorch"], default="legate"
)
_parser.add_argument("--master-addr", default=None)
_parser.add_argument("--master-port", type=int, default=29500)
_parser.add_argument("--legate-base-port", type=int, default=29600)
_args, _ = _parser.parse_known_args()

MASTER_ADDR = _args.master_addr or os.environ.get("MASTER_ADDR", "127.0.0.1")
TORCH_MASTER_PORT = str(_args.master_port)
LEGATE_BASE_PORT = _args.legate_base_port

# If PyTorch is the driver, set up Realm p2p bootstrap env vars BEFORE
# importing legate/cupynumeric (which triggers realm_init).
if _args.driver == "pytorch":
    import torch.distributed as dist

    dist.init_process_group(backend="gloo", init_method="env://")
    _rank = dist.get_rank()
    _world_size = dist.get_world_size()

    # Bind each torchrun process to its own GPU (torchrun sets LOCAL_RANK)
    _local_rank = int(os.environ.get("LOCAL_RANK", 0))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_local_rank)

    # Gather real IPs so p2p bootstrap works across nodes
    _my_ip = socket.gethostbyname(socket.gethostname())
    _all_ips = [None] * _world_size
    dist.all_gather_object(_all_ips, _my_ip)
    all_addrs = [
        f"{_all_ips[r]}:{LEGATE_BASE_PORT + r}" for r in range(_world_size)
    ]
    print(f"[rank {_rank}] p2p addrs: {all_addrs}")
    os.environ["WORKER_SELF_INFO"] = all_addrs[_rank]
    os.environ["WORKER_PEERS_INFO"] = " ".join(all_addrs)
    os.environ["BOOTSTRAP_P2P_PLUGIN"] = "realm_ucp_bootstrap_p2p.so"
    os.environ["REALM_UCP_BOOTSTRAP_MODE"] = "p2p"
    os.environ.setdefault(
        "LEGATE_CONFIG", "--cpus 1 --gpus 0 --min-cpu-chunk 1"
    )
    os.environ.setdefault("LEGATE_AUTO_CONFIG", "0")

import cupynumeric as cn  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from legate.core import (  # noqa: E402
    TaskContext,
    VariantCode,
    VariantOptions,
    get_legate_runtime,
)
from legate.core.task import task, OutputStore  # noqa: E402


# N is chosen to divide evenly across ranks except the last rank. Each rank r fills its shard of
# local_n = N / world_size elements with (arange(local_n) + r * 100) * 2.
# Verify the computation with implicit gather at top level.
N = 8


@task(
    variants=(VariantCode.CPU, VariantCode.GPU),
    options=VariantOptions(concurrent=True),
)
def legate_fill(ctx: TaskContext, dst: OutputStore) -> None:
    rank = get_legate_runtime().node_id

    t = torch.from_dlpack(dst)
    local_n = t.shape[0]
    print(
        f"[rank {rank}] device: {t.device}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}"
    )

    # Fill and double the local shard (zero-copy, writes into PhysicalStore)
    t[:] = (
        torch.arange(local_n, dtype=torch.float64, device=t.device)
        + rank * 100
    ) * 2
    print(f"[rank {rank}] shard after fill+double: {t.tolist()}")


def main():
    if _args.driver == "legate":
        rt = get_legate_runtime()
        if rt.node_count > 1:
            rank = rt.node_id
            world_size = rt.node_count
        else:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://{MASTER_ADDR}:{TORCH_MASTER_PORT}",
        )
    else:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    print(
        f"[rank {rank}] both runtimes initialized (driver={_args.driver}, world_size={world_size})"
    )

    # Legate creates a distributed store; task fills, doubles, and all_gathers
    data = cn.zeros(N, dtype=cn.float64)
    legate_fill(data)
    get_legate_runtime().issue_execution_fence(block=True)

    # Top-level access triggers implicit gather — should match all_gather
    t = torch.from_dlpack(data)
    print(f"[rank {rank}] implicit gather:         {t.tolist()}")

    # Verify: implicit gather matches expected doubled shards
    local_n = N // world_size
    shards = [
        (cn.arange(local_n, dtype=cn.float64) + r * 100) * 2
        for r in range(world_size)
    ]
    expected = cn.concatenate(shards)
    ok = bool(cn.allclose(cn.asarray(t), expected))
    print(f"[rank {rank}] result: {'pass' if ok else 'FAIL'}")

    dist.destroy_process_group()
    assert ok, "FAIL"


if __name__ == "__main__":
    main()
