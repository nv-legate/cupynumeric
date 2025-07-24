#!/usr/bin/env python3

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

from setuptools import find_packages
from skbuild import setup

import versioneer

setup(
    name="cupynumeric",
    version=versioneer.get_version(),
    description="An Aspiring Drop-In Replacement for NumPy at Scale",
    url="https://github.com/nv-legate/cupynumeric",
    author="NVIDIA Corporation",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Database",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    packages=find_packages(where=".", include=["cupynumeric*"]),
    package_data={"cupynumeric": ["_sphinxext/_templates/*.rst"]},
    include_package_data=True,
    cmdclass=versioneer.get_cmdclass(),
    install_requires=["cffi", "numpy>=1.22,!=2.1.0", "opt_einsum>=3.3"],
    zip_safe=False,
)
