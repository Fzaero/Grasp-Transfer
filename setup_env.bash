#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export REPO_DIR=${SCRIPT_DIR}
export GRASP_TRANSFER_SOURCE_DIR=${SCRIPT_DIR}/src/grasp_transfer