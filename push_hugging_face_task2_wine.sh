#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd "${SCRIPT_DIR}/task_2/hugging_face_spaces/wine" || { echo "Could not cd"; exit 1; }
git add .
git commit -m "Updated app at $(date)" || { echo "There's nothing new to commit."; exit 0; }
git push || { echo "Failed to push updates!"; exit 1; }