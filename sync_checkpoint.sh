#!/bin/bash
set -euo pipefail
RUN_DIR="./runs"
S3_BUCKET_NAME="training-s3-bucket-runpod/test"
REMOTE_FOLDER="$1"
# cmd
aws s3 sync "s3://${S3_BUCKET_NAME}/${REMOTE_FOLDER}" "${RUN_DIR}/${REMOTE_FOLDER}"