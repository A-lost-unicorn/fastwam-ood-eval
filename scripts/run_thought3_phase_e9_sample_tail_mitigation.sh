#!/usr/bin/env bash
set -euo pipefail

echo "Gate E.9a-v1 is archived as an invalid engineering run." >&2
echo "Do not resume or overwrite outputs/thought3/phase_e9_sample_tail_mitigation_v1." >&2
echo "Use scripts/run_thought3_phase_e9_sample_tail_mitigation_v2.sh after its clean preregistration commit." >&2
exit 2
