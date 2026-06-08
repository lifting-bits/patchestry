#!/usr/bin/env python3
# Copyright (c) 2025, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
"""Relay applied patches back to their binary addresses.

Given the patch-location map emitted by ``patchir-transform --emit-patch-map``
(the authoritative, apply-time source of truth), this tool resolves each applied
patch to a concrete address in the original ELF and confirms it against the
binary's disassembly. It emits a machine-readable JSON report and a
human-readable markdown table.

Address model
-------------
Ghidra loads the binary at its own image base, so the addresses stamped on the
CIR/LLVM-IR locations (e.g. ``ram:0002302c``) are *not* the ELF addresses. To
rebase robustly we never assume a global offset: for each patch we take the
in-function offset ``site - function_entry`` (both Ghidra addresses, from the
map) and add it to the function's ELF address resolved via ``nm``. This handles
PIE/rebased images per-function and sidesteps ARM-Thumb LSB tagging (offsets are
computed between even addresses).

Sources, in order of preference (``address_source`` in the output):
  * ``map``      -- the ``--map`` patchmap.json (primary; has function_address)
  * ``mlir_loc`` -- ``!mlir_loc`` metadata in the patched ``.ll`` (carries the
                    Ghidra address since patchir-transform now preserves it)
  * ``json``     -- Ghidra P-Code JSON call-target join (last-resort fallback)

Only the Python standard library is used; the binary side shells out to ``nm``
and ``objdump`` (mirroring verify_patched.sh).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _hex(value: int) -> str:
    return f"0x{value:x}"


def _parse_int(text: str) -> Optional[int]:
    """Parse a hex address that may or may not have a 0x prefix."""
    text = text.strip()
    if not text:
        return None
    try:
        return int(text, 16)
    except ValueError:
        return None


def _which_objdump() -> Optional[str]:
    for name in ("objdump", "llvm-objdump", "gobjdump"):
        path = shutil.which(name)
        if path:
            return path
    return None


# --------------------------------------------------------------------------
# Binary inspection (nm / objdump)
# --------------------------------------------------------------------------

class Binary:
    """Lazy nm/objdump views over an ELF, keyed by address."""

    def __init__(self, path: str):
        self.path = path
        self._symbols: Optional[dict[str, int]] = None
        self._disasm: Optional[dict[int, str]] = None

    def symbols(self) -> dict[str, int]:
        if self._symbols is None:
            self._symbols = {}
            nm = shutil.which("nm") or shutil.which("gnm")
            if not nm:
                print("warning: 'nm' not found; cannot resolve symbol addresses",
                      file=sys.stderr)
                return self._symbols
            out = subprocess.run(
                [nm, "--defined-only", self.path],
                capture_output=True, text=True, check=False,
            ).stdout
            for line in out.splitlines():
                parts = line.split()
                # Format: <addr> <type> <name>
                if len(parts) >= 3 and re.fullmatch(r"[0-9a-fA-F]+", parts[0]):
                    # Mask ARM-Thumb LSB so offsets land on the real address.
                    self._symbols.setdefault(parts[2], int(parts[0], 16) & ~1)
        return self._symbols

    def symbol_address(self, name: str) -> Optional[int]:
        return self.symbols().get(name)

    def disasm(self) -> dict[int, str]:
        if self._disasm is None:
            self._disasm = {}
            objdump = _which_objdump()
            if not objdump:
                print("warning: 'objdump' not found; cannot confirm disassembly",
                      file=sys.stderr)
                return self._disasm
            out = subprocess.run(
                [objdump, "-d", self.path],
                capture_output=True, text=True, check=False,
            ).stdout
            # Lines look like:  "   1302c: f7ef ecc8   blx  0x29c0 <sprintf@plt>"
            line_re = re.compile(r"^\s*([0-9a-fA-F]+):\s+(.*)$")
            for line in out.splitlines():
                m = line_re.match(line)
                if m:
                    self._disasm[int(m.group(1), 16)] = m.group(2).rstrip()
        return self._disasm

    def disasm_at(self, addr: int) -> Optional[str]:
        return self.disasm().get(addr)


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

@dataclass
class Located:
    patch: str = ""
    mode: str = ""
    function: str = ""
    callee: str = ""
    address_source: str = ""
    ghidra_site: str = ""          # Ghidra address of the matched site
    ghidra_function: str = ""      # Ghidra address of the enclosing function
    binary_address: str = ""       # resolved ELF address of the site
    disasm: str = ""               # objdump evidence at binary_address
    confirmed: bool = False        # disasm present and looks like a call/branch
    notes: list[str] = field(default_factory=list)


_CALL_RE = re.compile(r"\b(bl|blx|blr|call|b|bx|jal|jalr)\b", re.IGNORECASE)


def _resolve_elf_site(
    binary: Binary,
    func_name: str,
    ghidra_site: Optional[int],
    ghidra_func: Optional[int],
    rec: Located,
) -> Optional[int]:
    """Rebase a Ghidra site address onto the ELF via the in-function offset."""
    elf_func = binary.symbol_address(func_name)
    if elf_func is None:
        rec.notes.append(f"symbol '{func_name}' not found via nm")
        return None
    if ghidra_site is None or ghidra_func is None:
        rec.notes.append("missing Ghidra site/function address for rebase")
        return None
    offset = ghidra_site - ghidra_func
    if offset < 0:
        rec.notes.append(f"negative in-function offset {offset:#x}; suspect data")
    return elf_func + offset


def _confirm(binary: Binary, addr: Optional[int], rec: Located) -> None:
    if addr is None:
        return
    rec.binary_address = _hex(addr)
    line = binary.disasm_at(addr)
    if line is None:
        rec.notes.append("no disassembly at resolved address (rebase mismatch?)")
        return
    rec.disasm = line
    rec.confirmed = bool(_CALL_RE.search(line))
    if rec.callee and rec.callee in line:
        rec.confirmed = True


# --------------------------------------------------------------------------
# Primary path: the patch-location map
# --------------------------------------------------------------------------

def locate_from_map(map_path: str, binary: Binary) -> list[Located]:
    with open(map_path) as fh:
        data = json.load(fh)

    results: list[Located] = []
    for entry in data.get("patches", []):
        rec = Located(
            patch=entry.get("patch", ""),
            mode=entry.get("mode", ""),
            function=entry.get("function", ""),
            callee=entry.get("callee", ""),
            address_source="map",
            ghidra_site=entry.get("binary_address", ""),
            ghidra_function=entry.get("function_address", ""),
        )
        if not rec.ghidra_site:
            rec.notes.append("patchmap entry has no binary_address (unresolved at "
                             "apply time); check that the input CIR carried "
                             "ram: locations")
            results.append(rec)
            continue

        elf_site = _resolve_elf_site(
            binary, rec.function,
            _parse_int(rec.ghidra_site), _parse_int(rec.ghidra_function), rec,
        )
        _confirm(binary, elf_site, rec)
        results.append(rec)
    return results


# --------------------------------------------------------------------------
# Fallback path: patched .ll + (optional) Ghidra JSON
# --------------------------------------------------------------------------

_MD_LOC_RE = re.compile(r'^!(\d+) = !\{!"mlir_location", !"([^"]*)", i32 (\d+), i32 (\d+)\}')
_DEFINE_RE = re.compile(r"^define\b.*@([A-Za-z0-9_.$]+)\s*\(")
_PATCH_CALL_RE = re.compile(r"@(patch__[A-Za-z0-9_]+|__patchestry_[A-Za-z0-9_]+)\b")
_MLIR_LOC_USE_RE = re.compile(r"!mlir_loc !(\d+)")
_ADDR_KEY_RE = re.compile(r"^(\w+):([0-9a-fA-F]+)(?::\d+:\d+)?$")


def _ghidra_function_entries(json_path: Optional[str]) -> dict[str, int]:
    """Map function symbol name -> Ghidra entry address from the P-Code JSON.

    The JSON keys each function by its address (e.g. "ram:00022fe0"); the entry's
    "name"/"display_name" gives the symbol.
    """
    out: dict[str, int] = {}
    if not json_path:
        return out
    try:
        with open(json_path) as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return out
    for key, fn in data.get("functions", {}).items():
        m = _ADDR_KEY_RE.match(key)
        if not m:
            continue
        addr = int(m.group(2), 16)
        for field_name in ("display_name", "name"):
            name = fn.get(field_name)
            if name:
                out.setdefault(name, addr)
    return out


def _json_call_sites(json_path: Optional[str]) -> dict[str, list[tuple[str, int]]]:
    """Map function symbol -> [(callee_name, ghidra_site_addr), ...] from the JSON.

    A CALL op is keyed by its address ("ram:0002302c:157:6") and its
    ``target.function`` references another function entry whose ``name`` is the
    callee. Used to recover the matched site for a legacy ``.ll`` whose
    ``!mlir_loc`` predates location preservation.
    """
    out: dict[str, list[tuple[str, int]]] = {}
    if not json_path:
        return out
    try:
        with open(json_path) as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return out
    fns = data.get("functions", {})

    def callee_name(target: dict) -> str:
        ref = target.get("function")
        if ref and ref in fns:
            return fns[ref].get("name") or fns[ref].get("display_name") or ""
        return ""

    for fn in fns.values():
        fname = fn.get("display_name") or fn.get("name")
        if not fname:
            continue
        sites: list[tuple[str, int]] = []
        for bb in fn.get("basic_blocks", {}).values():
            for op_key, op in bb.get("operations", {}).items():
                if op.get("mnemonic") != "CALL":
                    continue
                m = _ADDR_KEY_RE.match(op_key)
                if not m:
                    continue
                sites.append((callee_name(op.get("target", {})), int(m.group(2), 16)))
        if sites:
            out[fname] = sites
    return out


def _derive_callee(patch_name: str) -> str:
    """Best-effort original callee from a patch symbol like patch__replace__sprintf."""
    m = re.match(r"patch__(?:replace|before|after|at_entrypoint)__(.+)", patch_name)
    return m.group(1) if m else ""


def locate_from_ll(ll_path: str, binary: Binary,
                   json_path: Optional[str]) -> list[Located]:
    loc_defs: dict[str, str] = {}          # "!N" -> mlir_location filename
    func_entries = _ghidra_function_entries(json_path)
    json_calls = _json_call_sites(json_path)

    with open(ll_path) as fh:
        lines = fh.readlines()

    # First pass: collect mlir_location metadata definitions.
    for line in lines:
        m = _MD_LOC_RE.match(line)
        if m:
            loc_defs[m.group(1)] = m.group(2)

    # Second pass: walk instructions, tracking the enclosing function.
    results: list[Located] = []
    current_func = ""
    for line in lines:
        dm = _DEFINE_RE.match(line)
        if dm:
            current_func = dm.group(1)
            continue
        call = _PATCH_CALL_RE.search(line)
        loc_use = _MLIR_LOC_USE_RE.search(line)
        if not call and not loc_use:
            continue
        if not (call or ("!patchestry" in line)):
            continue

        rec = Located(
            patch=call.group(1) if call else "",
            function=current_func,
            address_source="mlir_loc",
        )
        filename = loc_defs.get(loc_use.group(1), "") if loc_use else ""
        km = _ADDR_KEY_RE.match(filename) if filename else None
        if km:
            rec.ghidra_site = _hex(int(km.group(2), 16))
            ghidra_func = func_entries.get(current_func)
            if ghidra_func is not None:
                rec.ghidra_function = _hex(ghidra_func)
                rec.address_source = "mlir_loc+json" if json_path else "mlir_loc"
            elf_site = _resolve_elf_site(
                binary, current_func,
                _parse_int(rec.ghidra_site),
                ghidra_func, rec,
            )
            _confirm(binary, elf_site, rec)
        else:
            # Legacy .ll without a Ghidra address in !mlir_loc: recover the
            # matched site from the JSON by joining function + derived callee.
            resolved = False
            callee = _derive_callee(rec.patch)
            rec.callee = callee
            if callee and current_func in json_calls:
                matches = [addr for (cname, addr) in json_calls[current_func]
                           if cname == callee]
                if len(matches) == 1:
                    rec.address_source = "json"
                    rec.ghidra_site = _hex(matches[0])
                    ghidra_func = func_entries.get(current_func)
                    if ghidra_func is not None:
                        rec.ghidra_function = _hex(ghidra_func)
                    elf_site = _resolve_elf_site(
                        binary, current_func, matches[0], ghidra_func, rec)
                    _confirm(binary, elf_site, rec)
                    resolved = True
                elif len(matches) > 1:
                    rec.notes.append(
                        f"{len(matches)} calls to '{callee}' in {current_func}; "
                        "cannot disambiguate site without --map")
            if not resolved and not rec.notes:
                rec.address_source = "json" if json_path else "unresolved"
                rec.notes.append(
                    f"no Ghidra address in !mlir_loc (filename='{filename}') and "
                    f"no unique JSON call to '{callee or '?'}' in {current_func}; "
                    "rebuild with the location-preserving patchir-transform, or "
                    "supply --map")
        results.append(rec)
    return results


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def write_reports(results: list[Located], out_prefix: str,
                  binary_path: str, map_path: Optional[str]) -> None:
    payload = {
        "binary": binary_path,
        "source": map_path or "",
        "patches": [asdict(r) for r in results],
    }
    with open(out_prefix + ".json", "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")

    lines = [
        f"# Patch location report",
        "",
        f"- Binary: `{binary_path}`",
        f"- Patches located: {len(results)} "
        f"({sum(1 for r in results if r.confirmed)} confirmed in disassembly)",
        "",
        "| Patch | Mode | Function | Binary addr | Callee | Source | Evidence |",
        "|-------|------|----------|-------------|--------|--------|----------|",
    ]
    for r in results:
        evidence = r.disasm if r.confirmed else (r.disasm or "; ".join(r.notes))
        evidence = evidence.replace("|", "\\|")
        tick = "OK " if r.confirmed else "?? "
        lines.append(
            f"| {r.patch} | {r.mode} | {r.function} | "
            f"{r.binary_address or '-'} | {r.callee or '-'} | {r.address_source} | "
            f"{tick}{evidence} |"
        )
    lines.append("")
    with open(out_prefix + ".md", "w") as fh:
        fh.write("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Relay applied patches back to binary addresses.")
    ap.add_argument("--map", help="patchmap.json from patchir-transform "
                                   "--emit-patch-map (primary source)")
    ap.add_argument("--binary", required=True, help="original ELF binary")
    ap.add_argument("--ll", help="patched LLVM IR (.ll) fallback source")
    ap.add_argument("--json", dest="json_path",
                    help="Ghidra P-Code JSON (enables .ll-fallback rebase)")
    ap.add_argument("--out", default="patch_report",
                    help="output prefix for <prefix>.json and <prefix>.md")
    args = ap.parse_args()

    if not args.map and not args.ll:
        ap.error("provide --map (primary) and/or --ll (fallback)")

    binary = Binary(args.binary)

    results: list[Located] = []
    if args.map:
        results = locate_from_map(args.map, binary)
        # If the map produced no usable addresses, fall back to the .ll.
        if args.ll and not any(r.binary_address for r in results):
            print("note: patchmap yielded no addresses; falling back to --ll",
                  file=sys.stderr)
            results = locate_from_ll(args.ll, binary, args.json_path)
    else:
        results = locate_from_ll(args.ll, binary, args.json_path)

    write_reports(results, args.out, args.binary, args.map)

    confirmed = sum(1 for r in results if r.confirmed)
    print(f"Located {len(results)} patch(es), {confirmed} confirmed; "
          f"wrote {args.out}.json and {args.out}.md")
    for r in results:
        flag = "OK" if r.confirmed else "??"
        print(f"  [{flag}] {r.patch} ({r.mode}) -> {r.binary_address or 'UNRESOLVED'} "
              f"in {r.function}  [{r.address_source}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
