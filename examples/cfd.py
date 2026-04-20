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

# Text provided under a Creative Commons Attribution license, CC-BY.  All code is made available under the FSF-approved BSD-3 license.  (c) Lorena A. Barba, Gilbert F. Forsyth 2017. Thanks to NSF for support via CAREER award #1149784.

import argparse
import math

from _benchmark import benchmark_info, parse_with_harness


def build_up_b(np, package, rho, dt, dx, dy, u, v):
    b = np.zeros_like(u)
    if package == "legate":
        b.stencil_hint((2, 2), (2, 2))
    b[1:-1, 1:-1] = rho * (
        1
        / dt
        * (
            (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)
            + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)
        )
        - ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)) ** 2
        - 2
        * (
            (u[2:, 1:-1] - u[0:-2, 1:-1])
            / (2 * dy)
            * (v[1:-1, 2:] - v[1:-1, 0:-2])
            / (2 * dx)
        )
        - ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) ** 2
    )

    # Periodic BC Pressure @ x = 2
    b[1:-1, -1] = rho * (
        1
        / dt
        * (
            (u[1:-1, 0] - u[1:-1, -2]) / (2 * dx)
            + (v[2:, -1] - v[0:-2, -1]) / (2 * dy)
        )
        - ((u[1:-1, 0] - u[1:-1, -2]) / (2 * dx)) ** 2
        - 2
        * (
            (u[2:, -1] - u[0:-2, -1])
            / (2 * dy)
            * (v[1:-1, 0] - v[1:-1, -2])
            / (2 * dx)
        )
        - ((v[2:, -1] - v[0:-2, -1]) / (2 * dy)) ** 2
    )

    # Periodic BC Pressure @ x = 0
    b[1:-1, 0] = rho * (
        1
        / dt
        * (
            (u[1:-1, 1] - u[1:-1, -1]) / (2 * dx)
            + (v[2:, 0] - v[0:-2, 0]) / (2 * dy)
        )
        - ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx)) ** 2
        - 2
        * (
            (u[2:, 0] - u[0:-2, 0])
            / (2 * dy)
            * (v[1:-1, 1] - v[1:-1, -1])
            / (2 * dx)
        )
        - ((v[2:, 0] - v[0:-2, 0]) / (2 * dy)) ** 2
    )

    return b


def pressure_poisson_periodic(np, package, b, nit, p, dx, dy):
    pn = np.empty_like(p)

    for q in range(nit):
        pn = p.copy()
        if package == "legate":
            pn.stencil_hint((2, 2), (2, 2))
        p[1:-1, 1:-1] = (
            (pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2
            + (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2
        ) / (2 * (dx**2 + dy**2)) - dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[
            1:-1, 1:-1
        ]

        # Periodic BC Pressure @ x = 2
        p[1:-1, -1] = (
            (pn[1:-1, 0] + pn[1:-1, -2]) * dy**2
            + (pn[2:, -1] + pn[0:-2, -1]) * dx**2
        ) / (2 * (dx**2 + dy**2)) - dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[
            1:-1, -1
        ]

        # Periodic BC Pressure @ x = 0
        p[1:-1, 0] = (
            (pn[1:-1, 1] + pn[1:-1, -1]) * dy**2
            + (pn[2:, 0] + pn[0:-2, 0]) * dx**2
        ) / (2 * (dx**2 + dy**2)) - dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[
            1:-1, 0
        ]

        # Wall boundary conditions, pressure
        p[-1, :] = p[-2, :]  # dp/dy = 0 at y = 2
        p[0, :] = p[1, :]  # dp/dy = 0 at y = 0

    return p


@benchmark_info(name="CFD")
def channel_flow(
    np, package, iteration, warmup, nit, nx, ny, dt, dx, dy, rho, nu, *, timer
):
    # initial conditions
    u = np.zeros((ny, nx))
    un = np.zeros((ny, nx))

    v = np.zeros((ny, nx))
    vn = np.zeros((ny, nx))

    p = np.ones((ny, nx))

    b = np.zeros((ny, nx))

    print("Running channel flow...")

    stepcount = 0
    if package == "legate":
        u.stencil_hint((2, 2), (2, 2))
        v.stencil_hint((2, 2), (2, 2))
        p.stencil_hint((2, 2), (2, 2))

    timer.start()
    for _ in range(iteration + warmup):
        un = u.copy()

        vn = v.copy()
        if package == "legate":
            un.stencil_hint((2, 2), (2, 2))
            vn.stencil_hint((2, 2), (2, 2))

        b = build_up_b(np, package, rho, dt, dx, dy, u, v)
        p = pressure_poisson_periodic(np, package, b, nit, p, dx, dy)

        u[1:-1, 1:-1] = (
            un[1:-1, 1:-1]
            - un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2])
            - vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1])
            - dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2])
            + nu
            * (
                dt
                / dx**2
                * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2])
                + dt
                / dy**2
                * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])
            )
            + F * dt
        )
        v[1:-1, 1:-1] = (
            vn[1:-1, 1:-1]
            - un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2])
            - vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1])
            - dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1])
            + nu
            * (
                dt
                / dx**2
                * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2])
                + dt
                / dy**2
                * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])
            )
        )
        # Periodic BC u @ x = 2
        u[1:-1, -1] = (
            un[1:-1, -1]
            - un[1:-1, -1] * dt / dx * (un[1:-1, -1] - un[1:-1, -2])
            - vn[1:-1, -1] * dt / dy * (un[1:-1, -1] - un[0:-2, -1])
            - dt / (2 * rho * dx) * (p[1:-1, 0] - p[1:-1, -2])
            + nu
            * (
                dt / dx**2 * (un[1:-1, 0] - 2 * un[1:-1, -1] + un[1:-1, -2])
                + dt / dy**2 * (un[2:, -1] - 2 * un[1:-1, -1] + un[0:-2, -1])
            )
            + F * dt
        )
        # Periodic BC u @ x = 0
        u[1:-1, 0] = (
            un[1:-1, 0]
            - un[1:-1, 0] * dt / dx * (un[1:-1, 0] - un[1:-1, -1])
            - vn[1:-1, 0] * dt / dy * (un[1:-1, 0] - un[0:-2, 0])
            - dt / (2 * rho * dx) * (p[1:-1, 1] - p[1:-1, -1])
            + nu
            * (
                dt / dx**2 * (un[1:-1, 1] - 2 * un[1:-1, 0] + un[1:-1, -1])
                + dt / dy**2 * (un[2:, 0] - 2 * un[1:-1, 0] + un[0:-2, 0])
            )
            + F * dt
        )
        # Periodic BC v @ x = 2
        v[1:-1, -1] = (
            vn[1:-1, -1]
            - un[1:-1, -1] * dt / dx * (vn[1:-1, -1] - vn[1:-1, -2])
            - vn[1:-1, -1] * dt / dy * (vn[1:-1, -1] - vn[0:-2, -1])
            - dt / (2 * rho * dy) * (p[2:, -1] - p[0:-2, -1])
            + nu
            * (
                dt / dx**2 * (vn[1:-1, 0] - 2 * vn[1:-1, -1] + vn[1:-1, -2])
                + dt / dy**2 * (vn[2:, -1] - 2 * vn[1:-1, -1] + vn[0:-2, -1])
            )
        )
        # Periodic BC v @ x = 0
        v[1:-1, 0] = (
            vn[1:-1, 0]
            - un[1:-1, 0] * dt / dx * (vn[1:-1, 0] - vn[1:-1, -1])
            - vn[1:-1, 0] * dt / dy * (vn[1:-1, 0] - vn[0:-2, 0])
            - dt / (2 * rho * dy) * (p[2:, 0] - p[0:-2, 0])
            + nu
            * (
                dt / dx**2 * (vn[1:-1, 1] - 2 * vn[1:-1, 0] + vn[1:-1, -1])
                + dt / dy**2 * (vn[2:, 0] - 2 * vn[1:-1, 0] + vn[0:-2, 0])
            )
        )

        # Wall BC: u,v = 0 @ y = 0,2
        u[0, :] = 0

        u[-1, :] = 0
        v[0, :] = 0
        v[-1, :] = 0

        stepcount += 1

        if stepcount == warmup:
            sum_u = np.sum(u)
            sum_un = np.sum(un)
            udiff = (sum_u - sum_un) / sum_u  # ensure synchronization
            assert not math.isnan(udiff), (
                f"dt: {dt}. sum_u: {sum_u}. sum_un: {sum_un}. diff: {sum_u - sum_un}."
            )
            timer.start()

    sum_u = np.sum(u)
    sum_un = np.sum(un)
    udiff = (sum_u - sum_un) / sum_u  # ensure synchronization
    assert not math.isnan(udiff), (
        f"dt: {dt}. sum_u: {sum_u}. sum_un: {sum_un}. diff: {sum_u - sum_un}."
    )
    total = timer.stop()
    print(
        f"Elapsed Time: {total} ms. Steps: {stepcount - warmup}. "
        f"Warmups: {warmup}. dt: {dt}. udiff: {udiff}. "
        f"sum_u: {sum_u}. sum_un: {sum_un}. diff: {sum_u - sum_un}."
    )
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-x",
        "--nx",
        type=int,
        default=41,
        dest="nx",
        help="X-axis size of the grid",
    )
    parser.add_argument(
        "-y",
        "--ny",
        type=int,
        default=41,
        dest="ny",
        help="Y-axis size of the grid",
    )
    parser.add_argument(
        "-n",
        "--np",
        type=int,
        default=50,
        dest="np",
        help="Number of pseudo time steps to simulate in pressure Poisson",
    )
    parser.add_argument(
        "-i",
        "--iter",
        type=int,
        default=50,
        dest="iter",
        help="Number of iterations to run the simulation for",
    )
    parser.add_argument(
        "--dt",
        dest="dt",
        type=float,
        default=None,
        help="value of dt to use in simulation",
    )
    parser.add_argument(
        "--warmup",
        dest="warmup",
        type=int,
        default=1,
        help="number of iterations to warm up",
    )

    args, harness = parse_with_harness(parser)
    np = harness.np

    nx = args.nx
    ny = args.ny
    nit = args.np
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    dt = args.dt

    # physical variables
    rho = 1
    nu = 0.1
    F = 1
    if args.dt is None:
        sigma = 0.0009
        dt = sigma * dx * dy / nu
    else:
        dt = args.dt

    harness.run_timed(
        channel_flow,
        np,
        harness.package,
        args.iter,
        args.warmup,
        nit,
        nx,
        ny,
        dt,
        dx,
        dy,
        rho,
        nu,
        timer=harness.timer,
    )
