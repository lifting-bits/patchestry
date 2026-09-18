#!/usr/bin/env bash
#
# Lean-lift validation guard.
#
# For every fixture: lift with `-emit-flat-baseline -print-tu` (the 1:1 goto
# image of the P-Code), feed the printed C back through
# `-from-c -validate-pcode`, and require that it lowers to CIR with zero
# critical validation findings.  This pins two things at once: the printed C
# re-enters the tool, and the validator raises no false positives on the one
# body that is known to preserve every P-Code fact.
#
# Fixtures listed in flat-validate-known-failures.txt are expected to fail
# (their lifted AST is not valid C, or CIRGen cannot lower the re-parsed C); an unexpected failure or an unexpected pass
# fails the guard.
#
# Usage: check-flat-validate.sh <patchir-decomp> <strip-json-comments.sh> <fixture-dir>

set -u

decomp="$1"
strip="$2"
dir="$3"
known="$dir/flat-validate-known-failures.txt"

if [ ! -f "$known" ]; then
    echo "flat-validate: known-failures file not found: $known"
    exit 1
fi

work="$(mktemp -d "${TMPDIR:-/tmp}/flat-validate.XXXXXX")"
if [ -z "$work" ] || [ ! -d "$work" ]; then
    echo "flat-validate: cannot create a work directory"
    exit 1
fi
trap 'rm -rf "$work"' EXIT

is_known_failure() {
    sed 's/#.*//' "$known" | grep -qx "$1"
}

status=0
passed=0
known_failed=0
skipped=0
for json in "$dir"/*.json; do
    name="$(basename "$json" .json)"
    bash "$strip" "$json" > "$work/in.json" 2>/dev/null
    rm -f "$work/flat.c" "$work/rt.cir" "$work/rt.validation.json"
    "$decomp" -input "$work/in.json" -emit-flat-baseline -print-tu -output "$work/flat" \
        >/dev/null 2>&1
    if [ ! -s "$work/flat.c" ]; then
        skipped=$((skipped + 1))
        continue
    fi

    if "$decomp" -from-c "$work/flat.c" -input "$work/in.json" -validate-pcode -emit-cir \
        -output "$work/rt" > "$work/log.txt" 2>&1 && [ -s "$work/rt.cir" ]; then
        result=pass
    else
        result=fail
    fi

    if is_known_failure "$name"; then
        if [ "$result" = pass ]; then
            echo "FLAT-VALIDATE STALE ENTRY: $name now passes; remove it from flat-validate-known-failures.txt"
            status=1
        else
            known_failed=$((known_failed + 1))
        fi
    elif [ "$result" = fail ]; then
        echo "FLAT-VALIDATE REGRESSION: $name"
        grep -E "validate: |from-c: |Diag Error" "$work/log.txt" | head -5 | sed 's/^/    /'
        status=1
    else
        passed=$((passed + 1))
    fi
done

if [ "$passed" -eq 0 ] && [ "$known_failed" -eq 0 ]; then
    echo "flat-validate: no fixture produced C output; the decompiler or the environment is broken"
    exit 1
fi
if [ "$status" -eq 0 ]; then
    echo "flat-validate: OK ($passed validated, $known_failed known failures, $skipped without C output)"
fi
exit "$status"
