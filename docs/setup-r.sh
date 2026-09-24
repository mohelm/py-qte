#!/usr/bin/env sh
# Reproduce the pinned R environment for the benchmark docs.
#
#   1. installs the R version recorded in renv.lock (via rig, if available)
#   2. restores the exact package versions from the pinned CRAN snapshot
#
# Requires rig (https://github.com/r-lib/rig, optional) and network access.
# Usage: sh docs/setup-r.sh
set -eu

cd "$(dirname "$0")"
R_VERSION="$(sed -n 's/.*"Version": "\([0-9][0-9.]*\)".*/\1/p' renv.lock | head -n 1)"
REPO="$(sed -n 's/.*"URL": "\([^"]*\)".*/\1/p' renv.lock | head -n 1)"
echo "Pinned R version: ${R_VERSION}"
echo "CRAN snapshot:    ${REPO}"

R_CMD="Rscript"
if command -v rig >/dev/null 2>&1; then
  if ! rig list 2>/dev/null | grep -qw "${R_VERSION}"; then
    rig add "${R_VERSION}"
  fi
  R_CMD="rig run --r-version ${R_VERSION} --rscript"
fi

$R_CMD -e "if (!requireNamespace('renv', quietly = TRUE)) install.packages('renv', repos = '${REPO}'); renv::restore(project = '.', prompt = FALSE)"
