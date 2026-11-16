#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from elftools.elf.elffile import ELFFile
from patcherex2 import InsertFunctionPatch, ModifyFunctionPatch, Patcherex
from patcherex2.components.compilers import clang_arm as _patcherex_clang_arm
from patcherex2.targets.elf_arm_bare import ElfArmBare

import check_applied_bytes
import check_bindings
import check_trampoline
import patch_dependencies
import patch_record


# Patcherex2's ClangArm.compile() post-processes the assembled Thumb bytes and
# rewrites every `bl <known_symbol>` into `blx`. That is correct when patching
# an ARM-mode binary that wants to call into Thumb-compiled patches: the BLX
# (immediate) flips the encoding bit so the CPU exchanges modes on entry.
# For Cortex-M (Thumb-only) firmware, both the trampoline and the patch are
# Thumb, and the rewrite drops us into ARM on every patch call — the M-profile
# core has no ARM decoder and the firmware faults immediately. Skip the
# rewrite by short-circuiting the post-processor to the freshly compiled bytes.
def _thumb_only_compile(self, code, base=0, symbols=None, extension=".c",
                        extra_compiler_flags=None, is_thumb=False, **kwargs):
    if symbols is None:
        symbols = {}
    if extra_compiler_flags is None:
        extra_compiler_flags = []
    extra_compiler_flags = list(extra_compiler_flags)
    extra_compiler_flags += ["-mthumb"] if is_thumb else ["-mno-thumb"]
    # Skip ClangArm.compile entirely; defer to the base Compiler.compile so the
    # bl→blx (and blx→bl) rewrite never runs.
    return _patcherex_clang_arm.Compiler.compile(
        self, code, base=base, symbols=symbols, extension=extension,
        extra_compiler_flags=extra_compiler_flags, **kwargs,
    )


_patcherex_clang_arm.ClangArm.compile = _thumb_only_compile


@dataclass(frozen=True)
class Case:
    name: str
    function_name: str
    spec_name: str
    expected_lines: tuple[str, ...]
    # A problem check_trampoline must report for this case.  The case passes
    # when the rewritten ELF is rejected for that reason; it is not run.
    expect_reject: str | None = None


@dataclass(frozen=True)
class SymbolInfo:
    name: str
    addr: int
    size: int
    kind: str


CASES: tuple[Case, ...] = (
    Case(
        name="before",
        function_name="qemu_target_before",
        spec_name="serial_qemu_before_patch.yaml",
        expected_lines=(
            "BOOT",
            "PATCH:before",
            "BASE:before",
            "BASE:after",
            "BASE:replace",
            "BASE:contract",
            "BASE:short",
            "BASE:tiny",
            "BASE:pointer",
            "BASE:dispatch",
            "BASE:handler",
            "LEN:ok",
            "LEN:ok",
            "FWD:first",
            "FWD:second:small",
            "FWD:first",
            "FWD:second:large",
            "S",
            "z",
            "DONE",
        ),
    ),
    Case(
        name="after",
        function_name="qemu_target_after",
        spec_name="serial_qemu_after_patch.yaml",
        expected_lines=(
            "BOOT",
            "BASE:before",
            "BASE:after",
            "PATCH:after",
            "BASE:replace",
            "BASE:contract",
            "BASE:short",
            "BASE:tiny",
            "BASE:pointer",
            "BASE:dispatch",
            "BASE:handler",
            "LEN:ok",
            "LEN:ok",
            "FWD:first",
            "FWD:second:small",
            "FWD:first",
            "FWD:second:large",
            "S",
            "z",
            "DONE",
        ),
    ),
    Case(
        name="replace",
        function_name="qemu_target_replace",
        spec_name="serial_qemu_replace_patch.yaml",
        expected_lines=(
            "BOOT",
            "BASE:before",
            "BASE:after",
            "PATCH:replace",
            "BASE:contract",
            "BASE:short",
            "BASE:tiny",
            "BASE:pointer",
            "BASE:dispatch",
            "BASE:handler",
            "LEN:ok",
            "LEN:ok",
            "FWD:first",
            "FWD:second:small",
            "FWD:first",
            "FWD:second:large",
            "S",
            "z",
            "DONE",
        ),
    ),
    Case(
        name="contract",
        function_name="qemu_target_contract",
        spec_name="serial_qemu_entry_contract.yaml",
        expected_lines=(
            "BOOT",
            "BASE:before",
            "BASE:after",
            "BASE:replace",
            "CONTRACT:entry",
            "BASE:contract",
            "BASE:short",
            "BASE:tiny",
            "BASE:pointer",
            "BASE:dispatch",
            "BASE:handler",
            "LEN:ok",
            "LEN:ok",
            "FWD:first",
            "FWD:second:small",
            "FWD:first",
            "FWD:second:large",
            "S",
            "z",
            "DONE",
        ),
    ),
    # 4 bytes: the entry branch fills the function exactly.
    Case(
        name="short",
        function_name="qemu_target_short",
        spec_name="serial_qemu_short_patch.yaml",
        expected_lines=(
            "BOOT",
            "BASE:before",
            "BASE:after",
            "BASE:replace",
            "BASE:contract",
            "PATCH:short",
            "BASE:tiny",
            "BASE:pointer",
            "BASE:dispatch",
            "BASE:handler",
            "LEN:ok",
            "LEN:ok",
            "FWD:first",
            "FWD:second:small",
            "FWD:first",
            "FWD:second:large",
            "S",
            "z",
            "DONE",
        ),
    ),
    # 2 bytes: the entry branch overwrites the start of emit_tiny_base.
    # Rejected by the reinsertion check; the transcript is never compared.
    Case(
        name="tiny",
        function_name="qemu_target_tiny",
        spec_name="serial_qemu_tiny_patch.yaml",
        expected_lines=(
            "BOOT",
            "BASE:before",
            "BASE:after",
            "BASE:replace",
            "BASE:contract",
            "BASE:short",
            "PATCH:tiny",
            "BASE:pointer",
            "BASE:dispatch",
            "BASE:handler",
            "LEN:ok",
            "LEN:ok",
            "FWD:first",
            "FWD:second:small",
            "FWD:first",
            "FWD:second:large",
            "S",
            "z",
            "DONE",
        ),
        expect_reject="span-outside-function",
    ),
    # Reached only through an address stored in .rodata.
    Case(
        name="pointer",
        function_name="qemu_target_pointer",
        spec_name="serial_qemu_pointer_patch.yaml",
        expected_lines=(
            "BOOT",
            "BASE:before",
            "BASE:after",
            "BASE:replace",
            "BASE:contract",
            "BASE:short",
            "BASE:tiny",
            "PATCH:pointer",
            "BASE:dispatch",
            "BASE:handler",
            "LEN:ok",
            "LEN:ok",
            "FWD:first",
            "FWD:second:small",
            "FWD:first",
            "FWD:second:large",
            "S",
            "z",
            "DONE",
        ),
    ),
    # The callee of an indirect call: reached through qemu_callback.
    Case(
        name="handler",
        function_name="qemu_target_handler",
        spec_name="serial_qemu_handler_patch.yaml",
        expected_lines=(
            "BOOT",
            "BASE:before",
            "BASE:after",
            "BASE:replace",
            "BASE:contract",
            "BASE:short",
            "BASE:tiny",
            "BASE:pointer",
            "BASE:dispatch",
            "PATCH:handler",
            "LEN:ok",
            "LEN:ok",
            "FWD:first",
            "FWD:second:small",
            "FWD:first",
            "FWD:second:large",
            "S",
            "z",
            "DONE",
        ),
    ),
    # The caller: its lifted body reads qemu_callback and calls through it.
    Case(
        name="dispatch",
        function_name="qemu_target_dispatch",
        spec_name="serial_qemu_dispatch_patch.yaml",
        expected_lines=(
            "BOOT",
            "BASE:before",
            "BASE:after",
            "BASE:replace",
            "BASE:contract",
            "BASE:short",
            "BASE:tiny",
            "BASE:pointer",
            "PATCH:dispatch",
            "BASE:dispatch",
            "BASE:handler",
            "LEN:ok",
            "LEN:ok",
            "FWD:first",
            "FWD:second:small",
            "FWD:first",
            "FWD:second:large",
            "S",
            "z",
            "DONE",
        ),
    ),
    # Inserted code with a switch and string constants of its own; the
    # transcript shows every arm.
    Case(
        name="switch",
        function_name="qemu_target_replace",
        spec_name="serial_qemu_switch_patch.yaml",
        expected_lines=(
            "BOOT",
            "BASE:before",
            "BASE:after",
            "PATCH:switch:zero",
            "PATCH:switch:one",
            "PATCH:switch:two",
            "PATCH:switch:three",
            "PATCH:switch:four",
            "PATCH:switch:other",
            "BASE:contract",
            "BASE:short",
            "BASE:tiny",
            "BASE:pointer",
            "BASE:dispatch",
            "BASE:handler",
            "LEN:ok",
            "LEN:ok",
            "FWD:first",
            "FWD:second:small",
            "FWD:first",
            "FWD:second:large",
            "S",
            "z",
            "DONE",
        ),
    ),
    # Inserted code that calls a compiler runtime helper the firmware does not
    # have.  Rejected before the image is rewritten.
    Case(
        name="divide",
        function_name="qemu_target_replace",
        spec_name="serial_qemu_divide_patch.yaml",
        expected_lines=(
            "BOOT",
            "BASE:before",
            "BASE:after",
            "PATCH:divide",
            "BASE:contract",
            "BASE:short",
            "BASE:tiny",
            "BASE:pointer",
            "BASE:dispatch",
            "BASE:handler",
            "LEN:ok",
            "LEN:ok",
            "FWD:first",
            "FWD:second:small",
            "FWD:first",
            "FWD:second:large",
            "S",
            "z",
            "DONE",
        ),
        expect_reject="unresolved-dependency",
    ),
    # Inserted code with a writable variable of its own.  The reinsertion places
    # code and read-only data only, so the variable would have no storage.
    Case(
        name="state",
        function_name="qemu_target_replace",
        spec_name="serial_qemu_state_patch.yaml",
        expected_lines=(
            "BOOT",
            "BASE:before",
            "BASE:after",
            "PATCH:state",
            "BASE:contract",
            "BASE:short",
            "BASE:tiny",
            "BASE:pointer",
            "BASE:dispatch",
            "BASE:handler",
            "LEN:ok",
            "LEN:ok",
            "FWD:first",
            "FWD:second:small",
            "FWD:first",
            "FWD:second:large",
            "S",
            "z",
            "DONE",
        ),
        expect_reject="writable-state",
    ),
    # The same with a non-zero initial value, which would also need a startup
    # copy from flash.
    Case(
        name="initialized",
        function_name="qemu_target_replace",
        spec_name="serial_qemu_initialized_patch.yaml",
        expected_lines=(
            "BOOT",
            "BASE:before",
            "BASE:after",
            "PATCH:initialized",
            "BASE:contract",
            "BASE:short",
            "BASE:tiny",
            "BASE:pointer",
            "BASE:dispatch",
            "BASE:handler",
            "LEN:ok",
            "LEN:ok",
            "FWD:first",
            "FWD:second:small",
            "FWD:first",
            "FWD:second:large",
            "S",
            "z",
            "DONE",
        ),
        expect_reject="writable-state",
    ),
    # Validation corpus: input-restricting patches, one per way of rejecting.
    # Early return from a replacement body: a length over 16 gets the error value.
    Case(
        name="length",
        function_name="qemu_target_length",
        spec_name="serial_qemu_length_patch.yaml",
        expected_lines=(
            "BOOT",
            "BASE:before",
            "BASE:after",
            "BASE:replace",
            "BASE:contract",
            "BASE:short",
            "BASE:tiny",
            "BASE:pointer",
            "BASE:dispatch",
            "BASE:handler",
            "LEN:ok",
            "LEN:error",
            "FWD:first",
            "FWD:second:small",
            "FWD:first",
            "FWD:second:large",
            "S",
            "z",
            "DONE",
        ),
    ),
    # Skipped operation: the second call is not made for a value over 32.
    Case(
        name="forward",
        function_name="qemu_target_forward",
        spec_name="serial_qemu_forward_patch.yaml",
        expected_lines=(
            "BOOT",
            "BASE:before",
            "BASE:after",
            "BASE:replace",
            "BASE:contract",
            "BASE:short",
            "BASE:tiny",
            "BASE:pointer",
            "BASE:dispatch",
            "BASE:handler",
            "LEN:ok",
            "LEN:ok",
            "FWD:first",
            "FWD:second:small",
            "FWD:first",
            "S",
            "z",
            "DONE",
        ),
    ),
    # Call to a handler that does not return: the run ends at REJECT.
    Case(
        name="send",
        function_name="qemu_target_send",
        spec_name="serial_qemu_send_patch.yaml",
        expected_lines=(
            "BOOT",
            "BASE:before",
            "BASE:after",
            "BASE:replace",
            "BASE:contract",
            "BASE:short",
            "BASE:tiny",
            "BASE:pointer",
            "BASE:dispatch",
            "BASE:handler",
            "LEN:ok",
            "LEN:ok",
            "FWD:first",
            "FWD:second:small",
            "FWD:first",
            "FWD:second:large",
            "S",
            "REJECT",
        ),
    ),
)

BASELINE_LINES: tuple[str, ...] = (
    "BOOT",
    "BASE:before",
    "BASE:after",
    "BASE:replace",
    "BASE:contract",
    "BASE:short",
    "BASE:tiny",
    "BASE:pointer",
    "BASE:dispatch",
    "BASE:handler",
    "LEN:ok",
    "LEN:ok",
    "FWD:first",
    "FWD:second:small",
    "FWD:first",
    "FWD:second:large",
    "S",
    "z",
    "DONE",
)

PATCHEREX2_LOAD_OPTIONS = {"rebase_granularity": 0x1000}


def run(
    argv: Iterable[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        list(argv),
        cwd=cwd,
        env=env,
        check=False,
        capture_output=capture_output,
    )
    if result.returncode != 0:
        command = " ".join(str(arg) for arg in argv)
        stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
        stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
        details = "\n".join(part for part in (stdout, stderr) if part)
        raise RuntimeError(f"Command failed: {command}\n{details}".rstrip())
    return result


def normalize_transcript(output: bytes) -> tuple[str, ...]:
    text = output.decode("utf-8", errors="replace").replace("\r\n", "\n")
    return tuple(line for line in text.splitlines() if line)


def run_qemu(qemu_system_arm: Path, kernel: Path) -> tuple[str, ...]:
    # Compare against the firmware's UART output only. QEMU's own diagnostic
    # messages (e.g. "Timer with period zero, disabling" emitted by the
    # lm3s6965evb timer model when the firmware idles after DONE) go to stderr
    # and would otherwise leak into the transcript and break the comparison.
    argv = [
        str(qemu_system_arm),
        "-M",
        "lm3s6965evb",
        "-nographic",
        "-kernel",
        str(kernel),
    ]
    try:
        result = subprocess.run(argv, check=False, capture_output=True, timeout=2.0)
        stdout = result.stdout or b""
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or b""
    return normalize_transcript(stdout)


def load_symbols(path: Path) -> dict[str, SymbolInfo]:
    with path.open("rb") as f:
        elf = ELFFile(f)
        symtab = elf.get_section_by_name(".symtab")
        if symtab is None:
            raise RuntimeError(f"Missing .symtab in {path}")
        symbols: dict[str, SymbolInfo] = {}
        for symbol in symtab.iter_symbols():
            if not symbol.name:
                continue
            kind = symbol["st_info"]["type"]
            addr = int(symbol["st_value"])
            if kind == "STT_FUNC":
                addr &= ~1
            symbols[symbol.name] = SymbolInfo(
                name=symbol.name,
                addr=addr,
                size=int(symbol["st_size"]),
                kind=kind,
            )
        return symbols


def detect_arm_bare_regions(path: Path) -> tuple[int, int, int, int, int]:
    """Return (flash_start, flash_end, ram_start, ram_end, entry_point).

    PT_LOAD segments with PF_X and not PF_W are treated as flash; PF_W and
    not PF_X are treated as RAM. Each range is extended past the highest
    populated address so Patcherex2 has room to place inserted functions
    (1 MiB for flash, 64 KiB for RAM, matching the patche_firmware.py
    convention). Fail loudly if the heuristic returns no regions or the
    entry point is unset — silently substituting STM32-like defaults would
    miscompile firmwares with a different memory map (e.g. lm3s6965evb,
    where flash starts at 0x00000000).
    """
    with path.open("rb") as f:
        elf = ELFFile(f)
        flash_regions: list[tuple[int, int]] = []
        ram_regions: list[tuple[int, int]] = []
        for segment in elf.iter_segments():
            if segment["p_type"] != "PT_LOAD":
                continue
            vaddr = int(segment["p_vaddr"])
            end = vaddr + int(segment["p_memsz"])
            flags = int(segment["p_flags"])
            executable = bool(flags & 0x1)
            writable = bool(flags & 0x2)
            if executable and not writable:
                flash_regions.append((vaddr, end))
            elif writable and not executable:
                ram_regions.append((vaddr, end))
        entry_point = int(elf.header["e_entry"]) & ~1

    if not flash_regions:
        raise RuntimeError(
            f"{path}: no executable-only PT_LOAD segments found; cannot "
            "infer flash region for Patcherex2"
        )
    if not ram_regions:
        raise RuntimeError(
            f"{path}: no writable-only PT_LOAD segments found; cannot "
            "infer RAM region for Patcherex2"
        )
    if entry_point == 0:
        raise RuntimeError(f"{path}: ELF entry point is 0; cannot pick a Patcherex2 insert point")

    flash_start = min(start for start, _ in flash_regions)
    flash_end = max(end for _, end in flash_regions) + 0x100000
    ram_start = min(start for start, _ in ram_regions)
    ram_end = max(end for _, end in ram_regions) + 0x10000
    return flash_start, flash_end, ram_start, ram_end, entry_point


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def candidate_repo_roots(repo_root: Path) -> tuple[Path, ...]:
    candidates = [repo_root]
    if repo_root.parent.name == ".worktrees":
        candidates.append(repo_root.parent.parent)
    return tuple(candidates)


def repo_tool(repo_root: Path, build_type: str, tool_name: str) -> Path:
    # Try every builds/<preset>/ directory so this works for the local
    # `default` preset and CI's `ci` preset (and any other preset added later)
    # without needing to plumb a preset name from the caller.
    tried: list[Path] = []
    for root in candidate_repo_roots(repo_root):
        builds_dir = root / "builds"
        if not builds_dir.is_dir():
            continue
        # Prefer "default" first (legacy local-dev path), then any other preset.
        preset_dirs = sorted(
            (p for p in builds_dir.iterdir() if p.is_dir()),
            key=lambda p: (p.name != "default", p.name),
        )
        for preset_dir in preset_dirs:
            path = preset_dir / "tools" / tool_name / build_type / tool_name
            tried.append(path)
            if path.exists():
                return path
    searched = "\n  ".join(str(p) for p in tried) or "(no builds/ subdirectories found)"
    raise RuntimeError(
        f"Missing tool {tool_name} under any build tree for {repo_root}. "
        f"Build a preset that produces {build_type} artifacts first.\n"
        f"Searched:\n  {searched}"
    )


# Flags for the Patcherex2 compile of the patched LLVM IR.  The lifter emits
# IR with no type-based aliasing, no `inbounds` and wrapping signed arithmetic;
# the patch compiler must not add those assumptions back.  The optimization
# level is appended per run (`--patch-opt-level`).
#
# `-fno-pic`: Patcherex2 compiles for arm-linux-gnueabihf, where clang defaults
# to position-independent code and reaches an external global through a GOT
# slot.  Nothing creates that slot in a bare-metal image, so the inserted code
# would load the global's address from unmapped memory.  The image is linked at
# fixed addresses, so the address can be used directly.
PATCH_COMPILER_FLAGS = ("-fno-strict-aliasing", "-fwrapv", "-fno-pic")


# `Case.expect_reject` values for a case stopped before the image is rewritten:
# its inserted code references a symbol nothing provides, or owns a writable
# global.
UNRESOLVED_DEPENDENCY = "unresolved-dependency"
WRITABLE_STATE = "writable-state"


def patch_compiler_flags(opt_level: str) -> list[str]:
    return [*PATCH_COMPILER_FLAGS, f"-{opt_level}"]


def patch_function_in_elf(
    original_elf: Path,
    snippets: dict[str, str],
    target_function: str,
    rewritten_elf: Path,
    extra_compiler_flags: list[str],
    data_symbols: dict[str, int],
) -> tuple[list[patch_record.Applied], list[tuple[int, bytes]]]:
    """Reinsert the snippets; return what each patch wrote and what else was written."""
    if target_function not in snippets:
        raise RuntimeError(
            f"Patched LLVM IR does not define {target_function}. Defines: {sorted(snippets)}"
        )

    flash_start, flash_end, ram_start, ram_end, entry_point = detect_arm_bare_regions(original_elf)

    p = Patcherex(
        str(original_elf),
        target_cls=ElfArmBare,
        target_opts={
            "binary_analyzer": "angr",
            "compiler": "clang19",
            "binfmt_tool": "default",
            "allocation_manager": "default",
        },
        components_opts={
            "binfmt_tool": {
                "flash_start": flash_start,
                "flash_end": flash_end,
                "ram_start": ram_start,
                "ram_end": ram_end,
                # ElfArmBare.finalize() injects a save_context push/pop shim
                # at every insert_point to copy any FLASH-resident "new mapped
                # blocks" into RAM at boot. The shim uses force_insert=True
                # and clobbers the 4 bytes at the insert_point without
                # relocating the displaced instructions, so pointing it at
                # entry_point destroys Reset_Handler's .data-init prologue
                # and the firmware fails to reach main. None of the current
                # patches add RAM-resident blocks, so the shim has nothing to
                # do — pass an empty list to skip it. Re-introduce the entry
                # insert_point only if a patch lives in RAM and needs the
                # FLASH→RAM copy stub, and even then point it past the
                # .data/.bss init in startup.S, not at the reset vector.
                "insert_points": [],
            }
        },
    )

    # Patcherex2 resolves the patch's external references from the function
    # symbols angr finds.  It does not define data symbols, so a global the
    # lifted code reads or writes would be left unresolved and get whatever
    # address the patch loader invents.  Give it the address each lifted
    # global has in the original image.
    p.symbols.update(data_symbols)

    compile_opts = {
        "extension": ".ll",
        "load_options": PATCHEREX2_LOAD_OPTIONS,
        "extra_compiler_flags": list(extra_compiler_flags),
    }
    # Cortex-M / ElfArmBare targets only execute Thumb. InsertFunctionPatch
    # defaults is_thumb=False, which makes patcherex2 compile the patch with
    # -mno-thumb (ARM mode); the BLX from the Thumb trampoline then jumps
    # into ARM code and the M-profile core faults on the first instruction.
    # Force Thumb for every inserted helper.  `snippets` is in insertion order:
    # each helper is linked against the ones placed before it.
    recorder = patch_record.Recorder()
    for name, code in snippets.items():
        if name == target_function:
            continue
        patch = InsertFunctionPatch(name, code, is_thumb=True, compile_opts=compile_opts)
        recorder.watch(patch, name, "inserted", code)
        p.patches.append(patch)
    patch = ModifyFunctionPatch(
        target_function, snippets[target_function], compile_opts=compile_opts
    )
    recorder.watch(patch, target_function, "target", snippets[target_function])
    p.patches.append(patch)
    p.apply_patches()
    p.save_binary(str(rewritten_elf))
    return recorder.applied, recorder.unattributed(p)


def patch_compiler_info(compiler_flags: list[str], thumb: bool = True) -> dict:
    """The compiler the reinsertion runs, for the patch record."""
    # Patcherex2's "clang19" component runs `clang-19` from PATH.
    path = shutil.which("clang-19")
    version = None
    if path:
        result = subprocess.run([path, "--version"], capture_output=True, text=True, check=False)
        version = result.stdout.splitlines()[0] if result.stdout else None
    return {
        "path": str(Path(path).resolve()) if path else None,
        "version": version,
        "target": patch_dependencies.PATCH_TARGET,
        "flags": [*compiler_flags, "-mthumb" if thumb else "-mno-thumb"],
    }


def verify_patch_record(
    original_elf: Path,
    rewritten_elf: Path,
    target_function: str,
    applied: list[patch_record.Applied],
    unattributed: list[tuple[int, bytes]],
    compiler_flags: list[str],
    dependencies_report_path: Path,
    trampoline_report_path: Path,
    record_path: Path,
) -> dict[str, str]:
    """Write the record of what the rewriter did and check it against the image."""
    dependencies = json.loads(dependencies_report_path.read_text(encoding="utf-8"))
    references = {row["function"]: row["undefined"] for row in dependencies["functions"]}
    record = patch_record.build_record(
        original_elf,
        rewritten_elf,
        target_function,
        applied,
        patch_compiler_info(compiler_flags),
        references,
        unattributed,
    )
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    trampoline = json.loads(trampoline_report_path.read_text(encoding="utf-8"))
    return patch_record.check_record(record, rewritten_elf, trampoline)


def original_object_addresses(bindings_path: Path) -> dict[str, int]:
    """Linker name -> address for every global the lifted unit takes from the binary."""
    manifest = json.loads(bindings_path.read_text(encoding="utf-8"))
    return {
        binding.get("symbol", binding["name"]): int(binding["address"], 16)
        for binding in manifest.get("bindings", [])
        if binding["kind"] == "object" and binding["origin"] == "original" and binding.get("address")
    }


def verify_bindings(
    bindings_path: Path, original_elf: Path, rewritten_elf: Path | None, report_path: Path
) -> None:
    """Fail the case when a lifted name does not resolve to the original entity."""
    manifest = json.loads(bindings_path.read_text(encoding="utf-8"))
    report = check_bindings.check_bindings(manifest, original_elf, rewritten_elf)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    failed = check_bindings.failures(report, strict=False)
    if failed:
        details = ", ".join(
            f"{row['verdict']} {row['kind']} '{row['name']}' at {row.get('address', '?')}"
            for row in failed
        )
        raise RuntimeError(f"Binding check failed against {original_elf}: {details}")


def verify_dependencies(
    patched_ll: Path,
    provided: set[str],
    compiler_flags: list[str],
    target_function: str,
    report_path: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """Split the patched module per function and check what it depends on.

    Returns the snippets, in the order to insert them (helpers before their
    callers, the target last), and the problems found, keyed by verdict:

    - `writable-state`: the module defines a writable global.  The patch link
      places code and read-only data only, and the runtime passes no
      `insert_points`, so the global would have no storage in RAM and no
      startup copy or clearing.
    - `unresolved-dependency`: a symbol neither the original image nor an
      earlier snippet defines, or inserted functions that reference each other.
    """
    tools = patch_dependencies.Tools.from_environment()
    snippets = patch_dependencies.split_module(patched_ll, tools)
    report = patch_dependencies.check_snippets(snippets, provided, tools, compiler_flags)
    state = patch_dependencies.writable_state(patched_ll, tools, compiler_flags)
    report["writable_state"] = state
    order, stuck = patch_dependencies.insertion_order(report, target_function)
    report["insertion_order"] = order
    report["mutually_dependent"] = stuck
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    ordered = {name: snippets[name] for name in order}
    if target_function in snippets:
        ordered[target_function] = snippets[target_function]

    problems: dict[str, str] = {}
    if state:
        problems[WRITABLE_STATE] = (
            "the inserted code owns writable globals with no storage in RAM: "
            + patch_dependencies.describe_state(state)
        )
    owned = {row["name"] for row in state}
    missing = sorted(
        {name for name in patch_dependencies.failures(report) if name not in owned} | set(stuck)
    )
    if missing:
        problems[UNRESOLVED_DEPENDENCY] = (
            "the inserted code references symbols nothing provides: " + ", ".join(missing)
        )
    return ordered, problems


def verify_applied_bytes(
    record_path: Path,
    original_elf: Path,
    rewritten_elf: Path,
    snippets: dict[str, str],
    report_path: Path,
) -> list[str]:
    """Compare the rewritten image with a fresh compile of each recorded function."""
    record = json.loads(record_path.read_text(encoding="utf-8"))
    report = check_applied_bytes.check_applied_bytes(
        record, original_elf, rewritten_elf, snippets, patch_dependencies.Tools.from_environment()
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return check_applied_bytes.failures(report)


def verify_trampoline(
    original_elf: Path,
    rewritten_elf: Path,
    function: str,
    function_json: Path,
    report_path: Path,
) -> list[str]:
    """Check how the function was reinserted; return the problems found."""
    record = json.loads(function_json.read_text(encoding="utf-8"))
    try:
        report = check_trampoline.check_trampoline(original_elf, rewritten_elf, function, record)
    except LookupError as error:
        raise RuntimeError(str(error)) from error
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return check_trampoline.failures(report)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--build-type", default="Debug")
    parser.add_argument("--qemu-system-arm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--refresh-ghidra-fixtures", action="store_true")
    parser.add_argument(
        "--patch-opt-level",
        default="O0",
        choices=["O0", "O1", "O2", "Os", "Oz"],
        help="optimization level for the Patcherex2 compile of the patched IR",
    )
    args = parser.parse_args()
    compiler_flags = patch_compiler_flags(args.patch_opt_level)

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_dir = (
        args.fixture_dir.resolve()
        if args.fixture_dir is not None
        else (repo_root / "test" / "qemu-firmware-runtime" / "fixtures").resolve()
    )

    firmware_dir = repo_root / "firmwares" / "serial-qemu"
    firmware_elf = firmware_dir / "build" / "serial.elf"
    if not firmware_elf.exists():
        raise RuntimeError(f"Missing firmware ELF at {firmware_elf}. Run the firmware build first.")

    original_symbols = load_symbols(firmware_elf)

    patchir_decomp = repo_tool(repo_root, args.build_type, "patchir-decomp")
    patchir_transform = repo_tool(repo_root, args.build_type, "patchir-transform")
    patchir_cir2llvm = repo_tool(repo_root, args.build_type, "patchir-cir2llvm")
    patchir_yaml_parser = repo_tool(repo_root, args.build_type, "patchir-yaml-parser")

    baseline_lines = run_qemu(args.qemu_system_arm, firmware_elf)
    write_lines(output_dir / "baseline.log", baseline_lines)
    if baseline_lines != BASELINE_LINES:
        raise RuntimeError(
            f"Baseline transcript mismatch.\nExpected: {BASELINE_LINES}\nObserved: {baseline_lines}"
        )

    summary_rows = ["case\tstatus\tartifact"]
    statuses: dict[str, str] = {}

    for case in CASES:
        case_dir = output_dir / case.name
        case_dir.mkdir(parents=True, exist_ok=True)

        json_path = case_dir / f"{case.function_name}.json"
        cir_stem = case_dir / case.function_name
        cir_path = case_dir / f"{case.function_name}.cir"
        patched_cir_path = case_dir / f"{case.function_name}.patched.cir"
        patched_ll_path = case_dir / f"{case.function_name}.patched.ll"
        rewritten_elf_path = case_dir / "rewritten.elf"
        transcript_path = case_dir / "qemu.log"
        bindings_path = case_dir / "bindings.manifest.json"
        bindings_report_path = case_dir / "bindings.json"
        trampoline_report_path = case_dir / "trampoline.json"
        dependencies_report_path = case_dir / "dependencies.json"
        patch_record_path = case_dir / "patch_record.json"
        applied_bytes_path = case_dir / "applied_bytes.json"

        fixture_json_path = fixture_dir / f"{case.function_name}.json"
        if args.refresh_ghidra_fixtures:
            decompile_script = repo_root / "scripts" / "ghidra" / "decompile-headless.sh"
            run(
                [
                    "bash",
                    str(decompile_script),
                    "--input",
                    str(firmware_elf),
                    "--function",
                    case.function_name,
                    "--output",
                    str(json_path),
                ]
            )
            fixture_json_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(json_path, fixture_json_path)
        else:
            if not fixture_json_path.exists():
                raise RuntimeError(
                    f"Missing checked-in JSON fixture {fixture_json_path}. "
                    "Either provide fixtures with --fixture-dir or rerun with --refresh-ghidra-fixtures."
                )
            shutil.copyfile(fixture_json_path, json_path)

        run(
            [
                str(patchir_decomp),
                "-input",
                str(json_path),
                "-emit-cir",
                "-emit-bindings",
                str(bindings_path),
                "-output",
                str(cir_stem),
            ]
        )
        verify_bindings(bindings_path, firmware_elf, None, bindings_report_path)

        transform_dir = repo_root / "test" / "patchir-transform"
        run(
            [str(patchir_yaml_parser), case.spec_name, "--validate"],
            cwd=transform_dir,
        )
        run(
            [
                str(patchir_transform),
                str(cir_path),
                "-spec",
                str(transform_dir / case.spec_name),
                "-o",
                str(patched_cir_path),
            ]
        )
        run(
            [
                str(patchir_cir2llvm),
                "-S",
                str(patched_cir_path),
                "-o",
                str(patched_ll_path),
            ]
        )

        if case.function_name not in original_symbols:
            raise RuntimeError(f"Original ELF is missing function symbol {case.function_name}.")

        data_symbols = original_object_addresses(bindings_path)
        snippets, dependency_problems = verify_dependencies(
            patched_ll_path,
            # What the patch linker defines: the image's functions, and the
            # globals passed to it by address.  Other objects of the image are
            # not visible to it.
            {name for name, symbol in original_symbols.items() if symbol.kind == "STT_FUNC"}
            | set(data_symbols),
            compiler_flags,
            case.function_name,
            dependencies_report_path,
        )
        if dependency_problems:
            if case.expect_reject not in dependency_problems:
                raise RuntimeError(
                    f"Dependency check failed for {case.name}: "
                    f"{'; '.join(dependency_problems.values())} (see {dependencies_report_path})."
                )
            statuses[case.name] = f"rejected:{case.expect_reject}"
            summary_rows.append(f"{case.name}\t{statuses[case.name]}\t-")
            continue
        if case.expect_reject in (UNRESOLVED_DEPENDENCY, WRITABLE_STATE):
            raise RuntimeError(
                f"Case {case.name} should have been rejected for {case.expect_reject}; "
                f"the dependency check found no problem (see {dependencies_report_path})."
            )

        applied, unattributed = patch_function_in_elf(
            firmware_elf,
            snippets,
            case.function_name,
            rewritten_elf_path,
            compiler_flags,
            data_symbols,
        )
        verify_bindings(bindings_path, firmware_elf, rewritten_elf_path, bindings_report_path)

        problems = verify_trampoline(
            firmware_elf, rewritten_elf_path, case.function_name, json_path, trampoline_report_path
        )
        record_problems = verify_patch_record(
            firmware_elf,
            rewritten_elf_path,
            case.function_name,
            applied,
            unattributed,
            compiler_flags,
            dependencies_report_path,
            trampoline_report_path,
            patch_record_path,
        )
        if record_problems:
            raise RuntimeError(
                f"Patch record check failed for {case.name}: "
                f"{'; '.join(f'{kind}: {detail}' for kind, detail in record_problems.items())} "
                f"(see {patch_record_path})."
            )

        byte_problems = verify_applied_bytes(
            patch_record_path, firmware_elf, rewritten_elf_path, snippets, applied_bytes_path
        )
        if byte_problems:
            raise RuntimeError(
                f"Applied bytes check failed for {case.name}: {', '.join(byte_problems)} "
                f"(see {applied_bytes_path})."
            )

        if case.expect_reject is not None:
            if case.expect_reject not in problems:
                raise RuntimeError(
                    f"Case {case.name} should have been rejected for {case.expect_reject}; "
                    f"the reinsertion check reported {problems or 'no problem'} "
                    f"(see {trampoline_report_path})."
                )
            statuses[case.name] = f"rejected:{case.expect_reject}"
            summary_rows.append(f"{case.name}\t{statuses[case.name]}\t{rewritten_elf_path}")
            continue
        if problems:
            raise RuntimeError(
                f"Reinsertion check failed for {case.name}: {', '.join(problems)} "
                f"(see {trampoline_report_path})."
            )

        transcript_lines = run_qemu(args.qemu_system_arm, rewritten_elf_path)
        write_lines(transcript_path, transcript_lines)
        if transcript_lines != case.expected_lines:
            raise RuntimeError(
                f"Transcript mismatch for {case.name}.\nExpected: {case.expected_lines}\nObserved: {transcript_lines}"
            )

        statuses[case.name] = "pass"
        summary_rows.append(f"{case.name}\tpass\t{rewritten_elf_path}")

    write_lines(output_dir / "patch-compiler-flags.txt", compiler_flags)
    summary_tsv = output_dir / "summary.tsv"
    summary_md = output_dir / "summary.md"
    write_lines(summary_tsv, summary_rows)
    write_lines(
        summary_md,
        [
            "# QEMU Firmware Runtime Summary",
            "",
            f"Baseline: {output_dir / 'baseline.log'}",
            "",
            *[f"- `{case.name}`: {statuses[case.name]}" for case in CASES],
        ],
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
