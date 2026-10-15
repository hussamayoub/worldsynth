#!/bin/bash
# Run a Cosmos3 Transfer spec with Cosmos3-Nano and a memory watchdog (DGX Spark, unified memory).
#
# Usage: run_transfer.sh <spec>   e.g. run_transfer.sh multi_control   or   run_transfer.sh /path/to/my_spec.json
# A bare name refers to a cookbook spec in $TRANSFER_ROOT/specs. Output: $TRANSFER_ROOT/outputs/Cosmos3-Nano/<spec name>/
# Logs: ~/cosmos-work/run_<spec>.log (inference) and ~/cosmos-work/run_<spec>_mem.log (free memory every 10 s).
set -u
SPEC=$1
COSMOS_ROOT=${COSMOS_ROOT:-$HOME/cosmos-work/cosmos}
TRANSFER_ROOT=$COSMOS_ROOT/cookbooks/cosmos3/generator/transfer
MIN_FREE_GIB=${MIN_FREE_GIB:-8}  # stop the run before the machine runs out of memory

[[ $SPEC == *.json ]] && SPEC_PATH=$SPEC || SPEC_PATH=$TRANSFER_ROOT/specs/$SPEC.json
NAME=$(basename "$SPEC_PATH" .json)
LOG=$HOME/cosmos-work/run_$NAME.log
MEM=$HOME/cosmos-work/run_${NAME}_mem.log

cd "$COSMOS_ROOT/packages/cosmos3"
source .venv/bin/activate
export LD_LIBRARY_PATH= PYTHONNOUSERSITE=1
rm -f "$MEM"
start=$(date +%s)
python -m cosmos_framework.scripts.inference --parallelism-preset=latency \
  -i "$SPEC_PATH" -o "$TRANSFER_ROOT/outputs/Cosmos3-Nano/" \
  --checkpoint-path Cosmos3-Nano --seed 2026 > "$LOG" 2>&1 &
PID=$!
while kill -0 $PID 2>/dev/null; do
  avail=$(free -g | awk '/Mem:/{print $7}')
  echo "$(date +%T) available_GiB=$avail" >> "$MEM"
  if [ "$avail" -lt "$MIN_FREE_GIB" ]; then echo "$(date +%T) LOW MEMORY - stopping run" >> "$MEM"; kill $PID; fi
  sleep 10
done
wait $PID
status=$?
echo "exit $status after $(( $(date +%s) - start )) s" >> "$LOG"
echo "exit $status after $(( $(date +%s) - start )) s; log: $LOG"
