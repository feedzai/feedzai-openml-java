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

VERSION=v"$1"

# Compare version, when different remove the build folder
FILE=lightgbmlib_build/__version__
if [ -f "$FILE" ]; then
  OLD_VERSION=$(head -n 1 "$FILE")
  if [ "$VERSION" != "$OLD_VERSION" ]; then
    echo "Renaming the folder."
    rm -rf lightgbmlib_build
  fi
fi

# Make LightGBM if it doesn't exist
DIR=lightgbmlib_build
if [ ! -d "$DIR" ]; then
  echo "Entering the folder."
  cd make-lightgbm || return
  echo "Building LightGBM $VERSION"
  bash make.sh "$VERSION" || return
  echo "Exiting the folder."
  cd .. || return
  echo "Renaming the folder."
  mv make-lightgbm/build lightgbmlib_build || return
  echo "Finished!"
fi
