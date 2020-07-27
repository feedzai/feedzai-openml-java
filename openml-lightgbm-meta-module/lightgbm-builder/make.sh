#!/usr/bin/env bash

# Copyright (c) 2020 Feedzai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# @author Sheng Wang (sheng.wang@feedzai.com)

set -e

export LIGHTGBM_REPO_URL="$1" # This env var will be consumed by make.sh.
LIGHTGBM_COMMIT_REF="$2"
LIGHTGBMLIB_VERSION="$3"

echo "Building LightGBM $LIGHTGBM_COMMIT_REF as lightgbmlib $LIGHTGBMLIB_VERSION"
cd make-lightgbm
bash make.sh "$LIGHTGBM_COMMIT_REF" "$LIGHTGBMLIB_VERSION"

