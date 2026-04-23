#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from elftools.elf.elffile import ELFFile
from patcherex2 import ModifyRawBytesPatch, Patcherex
from patcherex2.targets import BinArmBare


@dataclass(frozen=True)
class Case:
    name: str
    function_name: str
    spec_name: str
    expected_lines: tuple[str, ...]


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
        spec_name="qemu_serial_before_patch.yaml",
        expected_lines=(
            "BOOT",
            "PATCH:before",
            "BASE:before",
            "BASE:after",
            "BASE:replace",
            "BASE:contract",
            "DONE",
        ),
    ),
    Case(
        name="after",
        function_name="qemu_target_after",
        spec_name="qemu_serial_after_patch.yaml",
        expected_lines=(
            "BOOT",
            "BASE:before",
            "BASE:after",
            "PATCH:after",
            "BASE:replace",
            "BASE:contract",
            "DONE",
        ),
    ),
    Case(
        name="replace",
        function_name="qemu_target_replace",
        spec_name="qemu_serial_replace_patch.yaml",
        expected_lines=(
            "BOOT",
            "BASE:before",
            "BASE:after",
            "PATCH:replace",
            "BASE:contract",
            "DONE",
        ),
    ),
    Case(
        name="contract",
        function_name="qemu_target_contract",
        spec_name="qemu_serial_entry_contract.yaml",
        expected_lines=(
            "BOOT",
            "BASE:before",
            "BASE:after",
            "BASE:replace",
            "CONTRACT:entry",
            "BASE:contract",
            "DONE",
        ),
    ),
)

BASELINE_LINES: tuple[str, ...] = (
    "BOOT",
    "BASE:before",
    "BASE:after",
    "BASE:replace",
    "BASE:contract",
    "DONE",
)

PATCH_ARENA_SLOT_SIZE = 0x100
THUMB_NOP = b"\x00\xbf"


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
        output = (result.stdout or b"") + (result.stderr or b"")
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or b"") + (exc.stderr or b"")
    return normalize_transcript(output)


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


def undefined_symbols(path: Path) -> list[str]:
    with path.open("rb") as f:
        elf = ELFFile(f)
        symtab = elf.get_section_by_name(".symtab")
        if symtab is None:
            raise RuntimeError(f"Missing .symtab in {path}")
        missing: list[str] = []
        for symbol in symtab.iter_symbols():
            if symbol["st_shndx"] != "SHN_UNDEF" or not symbol.name:
                continue
            missing.append(symbol.name)
        return missing


def dump_section(path: Path, section_name: str) -> bytes:
    with path.open("rb") as f:
        elf = ELFFile(f)
        section = elf.get_section_by_name(section_name)
        if section is None:
            raise RuntimeError(f"Missing {section_name} in {path}")
        return bytes(section.data())


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def candidate_repo_roots(repo_root: Path) -> tuple[Path, ...]:
    candidates = [repo_root]
    if repo_root.parent.name == ".worktrees":
        candidates.append(repo_root.parent.parent)
    return tuple(candidates)


def repo_tool(repo_root: Path, build_type: str, tool_name: str) -> Path:
    for root in candidate_repo_roots(repo_root):
        path = root / "builds" / "default" / "tools" / tool_name / build_type / tool_name
        if path.exists():
            return path
    raise RuntimeError(
        f"Missing tool {tool_name} under the default build tree for {repo_root}. "
        f"Build the {build_type} preset first."
    )


def compile_patched_module(
    llvm_clang: Path,
    patched_ll: Path,
    output_obj: Path,
) -> None:
    run(
        [
            str(llvm_clang),
            "--target=thumbv7m-none-eabi",
            "-mcpu=cortex-m3",
            "-mthumb",
            "-ffreestanding",
            "-fno-builtin",
            "-fno-stack-protector",
            "-Wno-override-module",
            "-c",
            str(patched_ll),
            "-o",
            str(output_obj),
        ]
    )


def link_patch_blob(
    ld_lld: Path,
    original_symbols: dict[str, SymbolInfo],
    input_obj: Path,
    patch_addr: int,
    output_elf: Path,
    linker_script: Path,
) -> None:
    linker_script.write_text(
        "\n".join(
            [
                "SECTIONS {",
                f"  .patchblob 0x{patch_addr:x} : ALIGN(4) {{",
                "    *(.text)",
                "    *(.rodata*)",
                "  }",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    argv = [
        str(ld_lld),
        str(input_obj),
        "-T",
        str(linker_script),
        "-o",
        str(output_elf),
    ]

    for name in undefined_symbols(input_obj):
        if name not in original_symbols:
            raise RuntimeError(f"Undefined symbol {name} from {input_obj} is not present in the original ELF.")
        symbol = original_symbols[name]
        value = symbol.addr | 1 if symbol.kind == "STT_FUNC" else symbol.addr
        argv.append(f"--defsym={name}=0x{value:x}")

    run(argv)


def patch_original_elf(
    original_elf: Path,
    rewritten_elf: Path,
    patch_blob: bytes,
    patch_addr: int,
    target_addr: int,
    target_size: int,
) -> None:
    p = Patcherex(str(original_elf), target_cls=BinArmBare)
    jump_bytes = p.assembler.assemble(f"b.w 0x{patch_addr:x}", target_addr, is_thumb=True)
    if len(jump_bytes) > target_size:
        raise RuntimeError(
            f"Detour jump for {hex(target_addr)} is {len(jump_bytes)} bytes, "
            f"which exceeds the original function size {target_size}."
        )
    remaining = target_size - len(jump_bytes)
    if remaining % len(THUMB_NOP) != 0:
        raise RuntimeError(
            f"Function at {hex(target_addr)} has size {target_size}, leaving a non-nop remainder {remaining}."
        )
    entry_patch = jump_bytes + THUMB_NOP * (remaining // len(THUMB_NOP))
    p.patches.append(ModifyRawBytesPatch(patch_addr, patch_blob))
    p.patches.append(ModifyRawBytesPatch(target_addr, entry_patch))
    p.apply_patches()
    p.save_binary(str(rewritten_elf))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--build-type", default="Debug")
    parser.add_argument("--llvm-prefix", type=Path, required=True)
    parser.add_argument("--ld-lld", type=Path, required=True)
    parser.add_argument("--qemu-system-arm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--refresh-ghidra-fixtures", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_dir = (
        args.fixture_dir.resolve()
        if args.fixture_dir is not None
        else (repo_root / "test" / "qemu-firmware-runtime" / "fixtures").resolve()
    )

    firmware_dir = repo_root / "firmwares" / "qemu-serial"
    firmware_elf = firmware_dir / "build" / "qemu-serial.elf"
    if not firmware_elf.exists():
        raise RuntimeError(f"Missing firmware ELF at {firmware_elf}. Run the firmware build first.")

    original_symbols = load_symbols(firmware_elf)
    for symbol_name in ("__patch_arena_start", "__patch_arena_end"):
        if symbol_name not in original_symbols:
            raise RuntimeError(f"Missing linker symbol {symbol_name} in {firmware_elf}.")

    patch_arena_start = original_symbols["__patch_arena_start"].addr
    patch_arena_end = original_symbols["__patch_arena_end"].addr
    required_patch_arena = patch_arena_start + PATCH_ARENA_SLOT_SIZE * len(CASES)
    if required_patch_arena > patch_arena_end:
        raise RuntimeError(
            f"Patch arena is too small: need {hex(required_patch_arena - patch_arena_start)} bytes, "
            f"have {hex(patch_arena_end - patch_arena_start)} bytes."
        )

    patchir_decomp = repo_tool(repo_root, args.build_type, "patchir-decomp")
    patchir_transform = repo_tool(repo_root, args.build_type, "patchir-transform")
    patchir_cir2llvm = repo_tool(repo_root, args.build_type, "patchir-cir2llvm")
    patchir_yaml_parser = repo_tool(repo_root, args.build_type, "patchir-yaml-parser")
    llvm_objcopy = args.llvm_prefix / "llvm-objcopy"
    llvm_clang = args.llvm_prefix / "clang"

    baseline_lines = run_qemu(args.qemu_system_arm, firmware_elf)
    write_lines(output_dir / "baseline.log", baseline_lines)
    if baseline_lines != BASELINE_LINES:
        raise RuntimeError(
            f"Baseline transcript mismatch.\nExpected: {BASELINE_LINES}\nObserved: {baseline_lines}"
        )

    summary_rows = ["case\tstatus\tartifact"]

    for index, case in enumerate(CASES):
        case_dir = output_dir / case.name
        case_dir.mkdir(parents=True, exist_ok=True)

        json_path = case_dir / f"{case.function_name}.json"
        cir_stem = case_dir / case.function_name
        cir_path = case_dir / f"{case.function_name}.cir"
        patched_cir_path = case_dir / f"{case.function_name}.patched.cir"
        patched_ll_path = case_dir / f"{case.function_name}.patched.ll"
        compiled_obj_path = case_dir / f"{case.function_name}.patched.o"
        linker_script_path = case_dir / "patch_blob.ld"
        patch_blob_elf_path = case_dir / "patch_blob.elf"
        rewritten_elf_path = case_dir / "rewritten.elf"
        transcript_path = case_dir / "qemu.log"

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

        run([str(patchir_decomp), "-input", str(json_path), "-emit-cir", "-output", str(cir_stem)])

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

        compile_patched_module(llvm_clang, patched_ll_path, compiled_obj_path)

        patch_addr = patch_arena_start + index * PATCH_ARENA_SLOT_SIZE
        link_patch_blob(
            args.ld_lld,
            original_symbols,
            compiled_obj_path,
            patch_addr,
            patch_blob_elf_path,
            linker_script_path,
        )

        patch_blob = dump_section(patch_blob_elf_path, ".patchblob")
        if len(patch_blob) > PATCH_ARENA_SLOT_SIZE:
            raise RuntimeError(
                f"Patch blob for {case.name} is {len(patch_blob)} bytes, "
                f"which exceeds slot size {PATCH_ARENA_SLOT_SIZE}."
            )

        target_symbol = original_symbols.get(case.function_name)
        if target_symbol is None:
            raise RuntimeError(f"Original ELF is missing function symbol {case.function_name}.")
        if target_symbol.size == 0:
            raise RuntimeError(f"Original function {case.function_name} has zero size in {firmware_elf}.")

        patch_original_elf(
            firmware_elf,
            rewritten_elf_path,
            patch_blob,
            patch_addr,
            target_symbol.addr,
            target_symbol.size,
        )

        transcript_lines = run_qemu(args.qemu_system_arm, rewritten_elf_path)
        write_lines(transcript_path, transcript_lines)
        if transcript_lines != case.expected_lines:
            raise RuntimeError(
                f"Transcript mismatch for {case.name}.\nExpected: {case.expected_lines}\nObserved: {transcript_lines}"
            )

        summary_rows.append(f"{case.name}\tpass\t{rewritten_elf_path}")

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
            *[f"- `{case.name}`: pass" for case in CASES],
        ],
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
