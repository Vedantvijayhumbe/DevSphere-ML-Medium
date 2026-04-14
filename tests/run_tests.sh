#!/usr/bin/env bash
# =============================================================
#  DevSphere ML-Medium — Test Runner
#
#  Called by:
#    - pre-commit hook (local)
#    - fork-ci.yml     (fork push, participant's CI minutes)
#    - pr-checks.yml   (PR to main, authoritative)
#
#  Platform support:
#    Linux / macOS / Windows (Git Bash or WSL)
#    Dependencies (TensorFlow, nbconvert, etc.) must be installed
#    before this script is called.  On CI this is handled by the
#    workflow's "Install dependencies" step.  Locally:
#
#      pip install -r requirements.txt
#
#  Exit 0 = evaluation passed minimum threshold
#  Exit 1 = notebook error, model missing, or below minimum score
# =============================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ── Run the evaluation pipeline ───────────────────────────────
#
#  evaluate.py handles everything:
#    1. Integrity check (notebook exists, uses MNIST, no pretrained models)
#    2. Notebook execution  (generates model.h5)
#    3. Model loading
#    4. Accuracy evaluation on test + unseen data
#    5. Score calculation and report
#
#  It exits 0 if score >= 11 (minimum threshold), 1 otherwise.
#  Output is printed to stdout for visibility in CI logs.

echo "── Digit Doctor — Evaluation Pipeline ──────────────────"
python3 evaluate.py
