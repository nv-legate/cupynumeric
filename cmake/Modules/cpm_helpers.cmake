#=============================================================================
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
#=============================================================================

function(get_cpm_git_args _out_var)

  set(oneValueArgs TAG BRANCH REPOSITORY)
  cmake_parse_arguments(GIT "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(repo_tag "")
  set(gh_tag_prefix "")
  # Default to specifying `GIT_REPOSITORY` and `GIT_TAG`
  set(cpm_git_args GIT_REPOSITORY ${GIT_REPOSITORY})

  if(GIT_BRANCH)
    set(gh_tag_prefix "heads")
    set(repo_tag "${GIT_BRANCH}")
    list(APPEND cpm_git_args GIT_TAG ${GIT_BRANCH})
  elseif(GIT_TAG)
    set(gh_tag_prefix "tags")
    set(repo_tag "${GIT_TAG}")
    list(APPEND cpm_git_args GIT_TAG ${GIT_TAG})
  endif()

  # Remove `.git` suffix from repo URL
  if(GIT_REPOSITORY MATCHES "^(.*)(\.git)$")
    set(GIT_REPOSITORY "${CMAKE_MATCH_1}")
  endif()
  if(GIT_REPOSITORY MATCHES "github\.com")
    # If retrieving from github use `.zip` URL to download faster
    set(cpm_git_args URL "${GIT_REPOSITORY}/archive/${repo_tag}.zip")
  elseif(GIT_REPOSITORY MATCHES "gitlab\.com")
    # GitLab archive URIs replace slashes with dashes
    string(REPLACE "/" "-" archive_tag "${repo_tag}")
    string(LENGTH "${GIT_REPOSITORY}" repo_name_len)
    string(FIND "${GIT_REPOSITORY}" "/" repo_name_idx REVERSE)
    math(EXPR repo_name_len "${repo_name_len} - ${repo_name_idx}")
    string(SUBSTRING "${GIT_REPOSITORY}" ${repo_name_idx} ${repo_name_len} repo_name)
    # If retrieving from gitlab use `.zip` URL to download faster
    set(cpm_git_args URL "${GIT_REPOSITORY}/-/archive/${repo_tag}/${repo_name}-${archive_tag}.zip")
  endif()

  set(${_out_var} ${cpm_git_args} PARENT_SCOPE)

endfunction()
