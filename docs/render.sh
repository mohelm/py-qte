#!/usr/bin/env sh
# Render the benchmark doc with the R version pinned in renv.lock.
#
# Uses a rig-managed R matching the pinned version when available, otherwise
# falls back to the R on PATH (and warns if the version differs). Quarto is
# pointed at the chosen R via QUARTO_R.
#
# Usage: sh docs/render.sh [-P key:value ...]
set -eu

here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
R_VERSION="$(sed -n 's/.*"Version": "\([0-9][0-9.]*\)".*/\1/p' "$here/renv.lock" | head -n 1)"

R_BIN=""
if command -v rig >/dev/null 2>&1 && rig list 2>/dev/null | grep -qw "$R_VERSION"; then
  R_BIN="$(rig run --r-version "$R_VERSION" --rscript \
    -e 'cat(file.path(R.home("bin"), "R"))' 2>/dev/null | tail -n 1)"
fi
[ -n "$R_BIN" ] || R_BIN="$(command -v R)"

ACTUAL="$("$R_BIN" --version 2>/dev/null | sed -n 's/^R version \([0-9.]*\).*/\1/p' | head -n 1)"
printf 'renv.lock R=%s | using R=%s (%s)\n' "$R_VERSION" "$ACTUAL" "$R_BIN"
[ "$ACTUAL" = "$R_VERSION" ] || printf 'WARNING: R version differs from renv.lock\n' >&2

cd "$root"
QUARTO_R="$R_BIN" uv run quarto render docs/aipw_benchmark.qmd "$@"
