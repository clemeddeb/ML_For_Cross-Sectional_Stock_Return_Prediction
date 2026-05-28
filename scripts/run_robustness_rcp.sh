#!/usr/bin/env bash
set -euo pipefail

# Submit the final robustness notebook run to EPFL RCP through RunAI.
#
# Required:
#   RUNAI_UID       Numeric RCP UID to pass to runai --run-as-uid.
#   RCP_USERNAME    Username owning the home PVC mount.
#
# Optional:
#   RUNAI_JOB_NAME  Defaults to ml-finance-robustness-analysis.
#   RUNAI_IMAGE     Defaults to registry.rcp.epfl.ch/ee559/environment-with-packages:latest.
#   RCP_REPO_DIR    Defaults to this repository path under /home/$RCP_USERNAME.
#   RUNAI_GPU       Defaults to 0 because the analysis only reads persisted artifacts.
#   PYTHON_BIN      Defaults to python3 from the RCP image.

RUNAI_UID="${RUNAI_UID:?Set RUNAI_UID to your numeric RCP UID.}"
RCP_USERNAME="${RCP_USERNAME:?Set RCP_USERNAME to your RCP username.}"

RUNAI_JOB_NAME="${RUNAI_JOB_NAME:-ml-finance-robustness-analysis}"
RUNAI_IMAGE="${RUNAI_IMAGE:-registry.rcp.epfl.ch/ee559/environment-with-packages:latest}"
RUNAI_GPU="${RUNAI_GPU:-0}"
RCP_REPO_DIR="${RCP_REPO_DIR:-/home/${RCP_USERNAME}/ML/ML_For_Finance_Project-AxelTurinPlessia-362559-ClementMeddeb-346164}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

REMOTE_COMMAND=$(cat <<EOF
set -euo pipefail
cd "${RCP_REPO_DIR}"
mkdir -p outputs/logs outputs/tables outputs/figures /tmp/matplotlib-cache
export MPLCONFIGDIR=/tmp/matplotlib-cache
"${PYTHON_BIN}" scripts/execute_notebook.py 2>&1 | tee outputs/logs/robustness_analysis_job.log
EOF
)

runai submit \
  --name "${RUNAI_JOB_NAME}" \
  --run-as-uid "${RUNAI_UID}" \
  --image "${RUNAI_IMAGE}" \
  --gpu "${RUNAI_GPU}" \
  --existing-pvc "claimname=home,path=/home/${RCP_USERNAME}" \
  --command -- bash -lc "${REMOTE_COMMAND}"
