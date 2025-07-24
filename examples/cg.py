#!/usr/bin/env python

# Copyright 2024 NVIDIA Corporation
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

import argparse

from benchmark import parse_args, run_benchmark


# This is technically dead code right now, but we'll keep it around in
# case we want to generate a symmetrix positive definite matrix later
def generate_random(N):
    print("Generating %dx%d symmetric positive definite system..." % (N, N))
    # Generate a random matrix
    A = np.random.rand(N, N)
    # Make sure the matrix is symmetric
    A = 0.5 * (A + A.T)
    # Now make sure that it is positive definite
    A = A + N * np.eye(N)
    b = np.random.rand(N)
    return A, b


def generate_2D(N, corners):
    if corners:
        print(
            "Generating %dx%d 2-D adjacency system with corners..."
            % (N**2, N**2)
        )
        A = np.zeros((N**2, N**2)) + 8 * np.eye(N**2)
    else:
        print(
            "Generating %dx%d 2-D adjacency system without corners..."
            % (N**2, N**2)
        )
        A = np.zeros((N**2, N**2)) + 4 * np.eye(N**2)
    # These are the same for both cases
    off_one = np.full(N**2 - 1, -1, dtype=np.float64)
    A += np.diag(off_one, k=1)
    A += np.diag(off_one, k=-1)
    off_N = np.full(N * (N - 1), -1, dtype=np.float64)
    A += np.diag(off_N, k=N)
    A += np.diag(off_N, k=-N)
    # If we have corners then we have four more cases
    if corners:
        off_N_plus = np.full(N * (N - 1) - 1, -1, dtype=np.float64)
        A += np.diag(off_N_plus, k=N + 1)
        A += np.diag(off_N_plus, k=-(N + 1))
        off_N_minus = np.full(N * (N - 1) + 1, -1, dtype=np.float64)
        A += np.diag(off_N_minus, k=N - 1)
        A += np.diag(off_N_minus, k=-(N - 1))
    # Then we can generate a random b matrix
    b = np.random.rand(N**2)
    return A, b


def check(A, x, b):
    print("Checking result...")
    if np.allclose(A.dot(x), b):
        print("PASS!")
    else:
        print("FAIL!")


def run_cg(
    N,
    corners,
    conv_iters,
    max_iters,
    warmup,
    conv_threshold,
    perform_check,
    timing,
    verbose,
):
    # A, b = generate_random(N)
    A, b = generate_2D(N, corners)

    print("Solving system...")
    x = np.zeros(A.shape[1])
    r = b - A.dot(x)
    p = r
    rsold = r.dot(r)
    converged = -1
    # Should always converge in fewer iterations than this
    max_iters = (
        min(max_iters, b.shape[0]) if max_iters is not None else b.shape[0]
    )

    timer.start()
    for i in range(-warmup, max_iters):
        if i == 0:
            timer.start()
        Ap = A.dot(p)
        alpha = rsold / (p.dot(Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.dot(r)
        # We only do the convergence test every conv_iters or on the last
        # iteration
        if (
            i >= 0
            and (i % conv_iters == 0 or i == (max_iters - 1))
            and np.sqrt(rsnew) < conv_threshold
        ):
            converged = i
            break
        if verbose:
            print("Residual: " + str(rsnew))
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
    total = timer.stop()

    if converged < 0:
        print("Convergence FAILURE!")
    else:
        print("Converged in %d iterations" % (converged))
    if perform_check:
        check(A, x, b)

    if timing:
        print(f"Elapsed Time: {total} ms")
    return total


def precondition(A, N, corners):
    if corners:
        d = 8 * (N**2)
    else:
        d = 4 * (N**2)
    M = np.diag(np.full(N**2, 1.0 / d))
    return M


def run_preconditioned_cg(
    N,
    corners,
    conv_iters,
    max_iters,
    warmup,
    conv_threshold,
    perform_check,
    timing,
    verbose,
):
    print("Solving system with preconditioner...")
    # A, b = generate_random(N)
    A, b = generate_2D(N, corners)
    M = precondition(A, N, corners)

    x = np.zeros(A.shape[1])
    r = b - A.dot(x)
    z = M.dot(r)
    p = z
    rzold = r.dot(z)
    converged = -1
    # Should always converge in fewer iterations than this
    max_iters = (
        min(max_iters, b.shape[0]) if max_iters is not None else b.shape[0]
    )

    timer.start()
    for i in range(-warmup, max_iters):
        if i == 0:
            timer.start()
        Ap = A.dot(p)
        alpha = rzold / (p.dot(Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        rznew = r.dot(r)
        # We only do the convergence test every conv_iters or on the
        # last iteration
        if (
            i >= 0
            and (i % conv_iters == 0 or i == (max_iters - 1))
            and np.sqrt(rznew) < conv_threshold
        ):
            converged = i
            break
        if verbose:
            print("Residual: " + str(rznew))
        z = M.dot(r)
        rznew = r.dot(z)
        beta = rznew / rzold
        p = z + beta * p
        rzold = rznew
    total = timer.stop()

    if converged < 0:
        print("Convergence FAILURE!")
    else:
        print("Converged in %d iterations" % (converged))
    if perform_check:
        check(A, x, b)

    if timing:
        print(f"Elapsed Time: {total} ms")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        dest="check",
        action="store_true",
        help="check the result of the solve",
    )
    parser.add_argument(
        "-k",
        "--corners",
        dest="corners",
        action="store_true",
        help="include corners when generating the adjacency matrix",
    )
    parser.add_argument(
        "-i",
        "--iters",
        type=int,
        default=25,
        dest="conv_iters",
        help="iterations between convergence tests",
    )
    parser.add_argument(
        "--pre",
        dest="precondition",
        action="store_true",
        help="use a Jacobi preconditioner",
    )
    parser.add_argument(
        "-m",
        "--max",
        type=int,
        default=None,
        dest="max_iters",
        help="bound the maximum number of iterations",
    )
    parser.add_argument(
        "-w",
        "--warmup",
        type=int,
        default=5,
        dest="warmup",
        help="warm-up iterations",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=10,
        dest="N",
        help="number of elements in one dimension",
    )
    parser.add_argument(
        "-t",
        "--time",
        dest="timing",
        action="store_true",
        help="perform timing",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="print verbose output",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-10,
        dest="conv_threshold",
        help="convergence check threshold",
    )

    args, np, timer = parse_args(parser)

    run_benchmark(
        run_preconditioned_cg if args.precondition else run_cg,
        args.benchmark,
        "PreCG" if args.precondition else "CG",
        (
            args.N,
            args.corners,
            args.conv_iters,
            args.max_iters,
            args.warmup,
            args.conv_threshold,
            args.check,
            args.timing,
            args.verbose,
        ),
    )
