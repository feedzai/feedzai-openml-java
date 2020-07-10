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

FILE=lightgbmlib_build
if [ ! -d "$FILE" ]; then
  echo "entering the folder."
  cd make-lightgbm || return
  echo "starting run the script"
  bash make.sh v2.3.0 || return
  echo "exiting the folder."
  cd .. || return
  echo "move the folder."
  mv make-lightgbm/build lightgbmlib_build || return
  echo "finished!"
fi
