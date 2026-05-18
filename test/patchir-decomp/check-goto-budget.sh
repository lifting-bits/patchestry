#!/usr/bin/env bash
#
# Goto-budget regression guard for the structuring cleanup pipeline.
#
# Runs patchir-decomp over every fixture in the directory and fails if any
# fixture emits more `goto` statements than its recorded budget.  This is a
# coarse, address-independent replacement for the per-goto `C-NOT:` FileCheck
# assertions: it pins the structuring KPI (total 25 gotos) without coupling
# the tests to specific address-derived label names.
#
# See docs/structuring_cleanup_redesign.md (Phase 0).
#
# Usage: check-goto-budget.sh <patchir-decomp> <strip-json-comments.sh> <fixture-dir>

set -u

decomp="$1"
strip="$2"
dir="$3"
baseline="$dir/goto-budget.txt"

if [ ! -f "$baseline" ]; then
    echo "goto-budget: baseline file not found: $baseline"
    exit 1
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

status=0
checked=0
for json in "$dir"/*.json; do
    name="$(basename "$json" .json)"
    bash "$strip" "$json" > "$work/in.json" 2>/dev/null
    "$decomp" -input "$work/in.json" -use-structuring-pass -print-tu \
        -output "$work/out" >/dev/null 2>&1
    emitted=0
    if [ -f "$work/out.c" ]; then
        emitted="$(grep -cE '\bgoto ' "$work/out.c" || true)"
    fi
    budget="$(awk -v n="$name" '$1 == n { print $2 }' "$baseline")"
    [ -z "$budget" ] && budget=0
    checked=$((checked + 1))
    if [ "$emitted" -gt "$budget" ]; then
        echo "GOTO BUDGET REGRESSION: $name emitted $emitted goto(s), budget $budget"
        status=1
    fi
    rm -f "$work/out.c"
done

if [ "$status" -eq 0 ]; then
    echo "goto-budget: OK ($checked fixtures within budget)"
fi
exit "$status"
