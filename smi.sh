#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$DIR"

export PASSIVE=0
export NOBOARD=1
export SIMULATION=1
export SKIP_FW_QUERY=1
export FINGERPRINT=TOYOTA_HIGHLANDER_TSS2
export PYTHONPATH="$DIR:$DIR/opendbc_repo:$DIR/msgq_repo:$DIR/rednose_repo:$DIR/teleoprtc_repo:$DIR/tinygrad_repo:${PYTHONPATH:-}"
export BLOCK="pandad,_pandad,camerad,loggerd,encoderd,micd,logmessaged${BLOCK:+,$BLOCK}"

python3 tools/sim/fake_panda.py &
FAKE_PANDA_PID=$!
trap 'kill "$FAKE_PANDA_PID" 2>/dev/null || true' EXIT

exec ./system/manager/manager.py
