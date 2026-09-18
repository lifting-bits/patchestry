#!/usr/bin/env bash
#
# Round-trip guard for the C printer.
#
# For every fixture in the directory: run `patchir-decomp -print-tu`, then
# re-parse the emitted C with `clang -fsyntax-only` under the fixture's target
# triple.  The printed C is the input of the LLM refinement stage and of
# `patchir-decomp -from-c`, so it has to be valid C.
#
# Fixtures listed in roundtrip-known-failures.txt are expected to fail.  An
# unexpected failure fails the guard; so does an unexpected pass, which means
# the entry is stale and must be removed.  The list therefore only shrinks,
# deliberately.
#
# Usage: check-roundtrip.sh <patchir-decomp> <strip-json-comments.sh> <clang> <fixture-dir>

set -u

decomp="$1"
strip="$2"
clang="$3"
dir="$4"
known="$dir/roundtrip-known-failures.txt"

if [ ! -f "$known" ]; then
    echo "roundtrip: known-failures file not found: $known"
    exit 1
fi

resource_dir="$("$clang" --print-resource-dir 2>/dev/null)"
if [ -z "$resource_dir" ]; then
    echo "roundtrip: cannot determine clang resource dir for $clang"
    exit 1
fi

work="$(mktemp -d "${TMPDIR:-/tmp}/roundtrip.XXXXXX")"
if [ -z "$work" ] || [ ! -d "$work" ]; then
    echo "roundtrip: cannot create a work directory"
    exit 1
fi
trap 'rm -rf "$work"' EXIT

# Map a Ghidra language id (`ARM:LE:32:Cortex`) to a clang triple.  Only the
# pointer width and endianness matter for a syntax-only parse.
triple_for() {
    local id="$1"
    local arch="${id%%:*}"
    local rest="${id#*:}"
    local endian="${rest%%:*}"
    rest="${rest#*:}"
    local bits="${rest%%:*}"
    case "$arch" in
        ARM)
            if [ "$endian" = "BE" ]; then echo "armebv7-unknown-linux-gnueabihf"
            else echo "armv7-unknown-linux-gnueabihf"; fi ;;
        AARCH64)
            if [ "$endian" = "BE" ]; then echo "aarch64_be-unknown-linux-gnu"
            else echo "aarch64-unknown-linux-gnu"; fi ;;
        x86)
            if [ "$bits" = "64" ]; then echo "x86_64-unknown-linux-gnu"
            else echo "i386-unknown-linux-gnu"; fi ;;
        MIPS)
            if [ "$bits" = "64" ]; then echo "mips64-unknown-linux-gnu"
            else echo "mips-unknown-linux-gnu"; fi ;;
        PowerPC)
            if [ "$bits" = "64" ]; then echo "powerpc64-unknown-linux-gnu"
            else echo "powerpc-unknown-linux-gnu"; fi ;;
        *)
            echo "armv7-unknown-linux-gnueabihf" ;;
    esac
}

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
    rm -f "$work/out.c"
    "$decomp" -input "$work/in.json" -print-tu -output "$work/out" >/dev/null 2>&1
    if [ ! -s "$work/out.c" ]; then
        # Negative fixtures produce no C; nothing to round-trip.
        skipped=$((skipped + 1))
        continue
    fi

    id="$(grep -o '"id" *: *"[^"]*"' "$work/in.json" | head -1 | sed 's/.*: *"//; s/"$//')"
    triple="$(triple_for "$id")"
    if "$clang" -fsyntax-only -x c -std=gnu23 -target "$triple" -nostdinc \
        -isystem "$resource_dir/include" -Wno-everything "$work/out.c" \
        > "$work/diag.txt" 2>&1; then
        result=pass
    else
        result=fail
    fi

    if is_known_failure "$name"; then
        if [ "$result" = pass ]; then
            echo "ROUNDTRIP STALE ENTRY: $name now re-parses; remove it from roundtrip-known-failures.txt"
            status=1
        else
            known_failed=$((known_failed + 1))
        fi
    elif [ "$result" = fail ]; then
        echo "ROUNDTRIP REGRESSION: $name does not re-parse ($triple):"
        grep -m 3 "error:" "$work/diag.txt" | sed 's/^/    /'
        status=1
    else
        passed=$((passed + 1))
    fi
done

if [ "$passed" -eq 0 ] && [ "$known_failed" -eq 0 ]; then
    echo "roundtrip: no fixture produced C output; the decompiler or the environment is broken"
    exit 1
fi
if [ "$status" -eq 0 ]; then
    echo "roundtrip: OK ($passed re-parsed, $known_failed known failures, $skipped without C output)"
fi
exit "$status"
