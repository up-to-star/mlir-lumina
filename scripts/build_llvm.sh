#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$(cd "$SCRIPT_DIR/../install" && pwd)"
LLVM_DIR="$(cd "$SCRIPT_DIR/../llvm-project" && pwd)"

cd ${LLVM_DIR}/build

cmake -G Ninja ../llvm \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;X86" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON

cd ${LLVM_DIR}/build
ninja install