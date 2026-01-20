#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$(cd "$SCRIPT_DIR/../install" && pwd)"
MIRA_DIR="$(cd "$SCRIPT_DIR/../mira" && pwd)"

cd ${MIRA_DIR}/build

cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}

ninja
