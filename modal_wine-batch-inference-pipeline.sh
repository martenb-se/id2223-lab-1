#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TASK="task_2"
NICE_LOG="nice_log.py"
PYTHON_SCRIPT=wine-batch-inference-pipeline.py

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 --local|--remote-run|--remote-deploy"
    exit 1
fi

# Check if the argument is valid
if [ "$1" != "--local" ] && [ "$1" != "--remote-run" ] && [ "$1" != "--remote-deploy" ]; then
    echo "Error: Invalid argument. Use --local, --remote-run or --remote-deploy."
    exit 1
fi

# ~~~~~~~ Colors
source "$SCRIPT_DIR/bash/colors.sh" || { echo "Could not source colors.sh"; exit 1; }

# ~~~~~~~ Modal Help
source "$SCRIPT_DIR/bash/modal.sh" || { echo "Could not source modal.sh"; exit 1; }

# Copy NiceLog temporarily
cp "${SCRIPT_DIR}/${NICE_LOG}" "${SCRIPT_DIR}/${TASK}/${NICE_LOG}"

# Run
(
  cd "${SCRIPT_DIR}/${TASK}" || { echo "Failed to go into project dir"; exit 1; }
  if [ "$1" == "--local" ]; then
    python $PYTHON_SCRIPT
    # script -q -c "python $PYTHON_SCRIPT" /dev/null
  elif [ "$1" == "--remote-run" ]; then
    modal_execute run
  elif [ "$1" == "--remote-deploy" ]; then
    modal_execute deploy
  fi
)

# Remove temporary NiceLog copy
rm "${SCRIPT_DIR}/${TASK}/${NICE_LOG}"
