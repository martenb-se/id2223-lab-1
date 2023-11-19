#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 --local|--remote"
    exit 1
fi

# Check if the argument is valid
if [ "$1" != "--local" ] && [ "$1" != "--remote" ]; then
    echo "Error: Invalid argument. Use --local or --remote."
    exit 1
fi


# Copy NiceLog temporarily
cp "${SCRIPT_DIR}/nice_log.py" "${SCRIPT_DIR}/task_1/nice_log.py"

# Run
(
  cd "${SCRIPT_DIR}/task_1" || { echo "Failed to go into project dir"; exit 1; }
  if [ "$1" == "--local" ]; then
    python iris-feature-pipeline-daily.py
  elif [ "$1" == "--remote" ]; then
    modal deploy iris-feature-pipeline-daily.py || {
      modal token new || { echo "Failed to get token"; exit 1; }
      modal deploy iris-feature-pipeline-daily.py
    }
  fi
)

# Remove temporary NiceLog copy
rm "${SCRIPT_DIR}/task_1/nice_log.py"
