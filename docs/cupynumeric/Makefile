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

# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS     ?= -v
PARALLEL_BUILD ?= 1
SPHINXBUILD    ?= $(shell which sphinx-build)
SOURCEDIR      = source
BUILDDIR       = build

.PHONY: help clean html linkcheck

help:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo "  all              to clean build standalone HTML files and run linkcheck"
	@echo "  clean            to clear all built documentation files"
	@echo "  html             to make standalone (non-gallery) HTML files"
	@echo "  linkcheck        to run the link checker on built docs"

all: clean html linkcheck

clean:
	rm -rf build/html/docs
	find source/ -name "generated" | xargs rm -rf

html:
	@start=$$(date +%s) \
	; LEGATE_AUTO_CONFIG=0 $(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) -j $(PARALLEL_BUILD) \
	&& mkdir -p build/html/docs \
	&& cp -r ../figures build/html/docs/ \
	&& cp switcher.json build/html/ \
	&& echo "\nBuild finished in $$(($$(date +%s)-start)) seconds\n"

linkcheck: html
	@start=$$(date +%s) \
	; $(SPHINXBUILD) "$(SOURCEDIR)" "$(BUILDDIR)" -b linkcheck \
	; echo "\nLink check finished in $$(($$(date +%s)-start)) seconds. Any broken links are listed below:\n" \
	&& [ -f "$(BUILDDIR)/output.txt" ] && cat "$(BUILDDIR)/output.txt"

docserve:
	python -m http.server -d build/html/
