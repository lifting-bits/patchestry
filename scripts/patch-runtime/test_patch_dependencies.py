#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Tests for patch_dependencies.py.

Needs clang, llvm-extract, opt, llvm-as and llvm-nm from PATCHESTRY_LLVM_PREFIX
(a bin directory), the variable scripts/test-qemu-firmware-runtime.sh uses.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import patch_dependencies  # noqa: E402

# f and g are the functions to insert.  They share one string and one writable
# counter; only f uses the table; nothing uses @unused.
MODULE = r"""
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7-unknown-linux-gnueabihf"

@.str = private unnamed_addr constant [4 x i8] c"one\00", align 1
@.str.1 = private unnamed_addr constant [4 x i8] c"two\00", align 1
@.str.2 = private unnamed_addr constant [6 x i8] c"three\00", align 1
@table = internal constant [2 x ptr] [ptr @.str, ptr @.str.1], align 4
@unused = internal constant i32 7, align 4
@counter = internal global i32 3, align 4
@limit = dso_local local_unnamed_addr global i32 9, align 4
@ext = external global i32, align 4

declare void @uart_write_line(ptr)
declare void @sink(ptr noundef, ptr noundef) #0

define dso_local void @f(i32 %i) #1 {
  %p = getelementptr [2 x ptr], ptr @table, i32 0, i32 %i
  %s = load ptr, ptr %p
  call void @uart_write_line(ptr %s), !note !1
  %c = load i32, ptr @counter
  %n = add i32 %c, 1
  store i32 %n, ptr @counter
  call void @g()
  ret void
}

define internal void @g() #1 {
  %c = load i32, ptr @counter
  %l = load i32, ptr @limit
  %e = load i32, ptr @ext
  call void @uart_write_line(ptr @.str.1)
  call void @uart_write_line(ptr @.str.2)
  call void @sink(ptr @.str.2, ptr @f)
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { noinline nounwind optnone }
!1 = !{!"note"}
!2 = !{!"never referenced"}
"""

# 64-bit division has no instruction on 32-bit ARM; the compiler calls a
# runtime helper for it.
NEEDS_HELPER = r"""
target triple = "armv7-unknown-linux-gnueabihf"

@.str = private unnamed_addr constant [4 x i8] c"div\00", align 1

declare void @uart_write_line(ptr)

define dso_local i64 @divide(i64 %a, i64 %b) {
  call void @uart_write_line(ptr @.str)
  %q = udiv i64 %a, %b
  %r = call i64 @local_helper(i64 %q)
  ret i64 %r
}

define internal i64 @local_helper(i64 %v) noinline {
  ret i64 %v
}
"""


def find_tools() -> patch_dependencies.Tools:
    prefix = os.environ.get("PATCHESTRY_LLVM_PREFIX")
    if not prefix:
        raise unittest.SkipTest("set PATCHESTRY_LLVM_PREFIX to the LLVM bin directory")
    tools = patch_dependencies.Tools.from_prefix(Path(prefix))
    missing = tools.missing()
    if missing:
        raise unittest.SkipTest(f"missing LLVM tools: {', '.join(missing)}")
    return tools


def definitions(snippet: str) -> dict[str, str]:
    """Global name -> its line, for every global the snippet defines."""
    found = {}
    for line in snippet.splitlines():
        match = re.match(r'^@("[^"]+"|[\w.$-]+) = (.*)$', line)
        if match and not re.match(r"(external|extern_weak)\b", match.group(2)):
            found[match.group(1)] = line
    return found


def declarations(snippet: str) -> set[str]:
    return {
        match.group(1)
        for match in re.finditer(r'^@("[^"]+"|[\w.$-]+) = external\b', snippet, re.MULTILINE)
    }


def defined_functions(snippet: str) -> list[str]:
    return re.findall(r'^define\b[^@]*@("[^"]+"|[\w.$-]+)\(', snippet, re.MULTILINE)


class SplitModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = find_tools()
        cls.directory = Path(tempfile.mkdtemp(prefix="patch-dependencies-"))
        cls.module = cls.directory / "module.ll"
        cls.module.write_text(MODULE, encoding="utf-8")
        cls.snippets = patch_dependencies.split_module(cls.module, cls.tools)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory, ignore_errors=True)

    def test_one_snippet_per_defined_function(self):
        self.assertEqual(sorted(self.snippets), ["f", "g"])

    def test_snippet_defines_only_its_own_function(self):
        self.assertEqual(defined_functions(self.snippets["f"]), ["f"])
        self.assertEqual(defined_functions(self.snippets["g"]), ["g"])
        # The other one is still declared, so the call and the address-of link.
        self.assertRegex(self.snippets["f"], r"(?m)^declare\b.*@g\(")
        self.assertRegex(self.snippets["g"], r"(?m)^declare\b.*@f\(")

    def test_reachable_constants_keep_their_initializers(self):
        f = definitions(self.snippets["f"])
        # The table and, through its initializer, the two strings it points to.
        self.assertIn("table", f)
        self.assertIn("ptr @.str, ptr @.str.1", f["table"])
        self.assertIn('c"one\\00"', f[".str"])
        self.assertIn('c"two\\00"', f[".str.1"])
        g = definitions(self.snippets["g"])
        self.assertIn('c"two\\00"', g[".str.1"])
        self.assertIn('c"three\\00"', g[".str.2"])

    def test_unreachable_constants_are_left_out(self):
        f = self.snippets["f"]
        g = self.snippets["g"]
        self.assertNotIn("@unused", f)
        self.assertNotIn("@unused", g)
        self.assertNotIn("@.str.2", f)  # only g prints "three"
        self.assertNotIn("@table", g)   # only f indexes the table
        self.assertNotIn('c"one\\00"', g)

    def test_constants_stay_local_to_the_snippet(self):
        # Each snippet is linked on its own; a constant must not become a
        # symbol the next snippet's copy could clash with or bind to.
        for name in ("f", "g"):
            for line in definitions(self.snippets[name]).values():
                self.assertRegex(line, r"= (private|internal)\b", line)

    def test_writable_globals_are_declared_not_copied(self):
        # A copy per snippet would give f and g a counter each.
        for name in ("f", "g"):
            self.assertNotIn("counter", definitions(self.snippets[name]))
            self.assertIn("counter", declarations(self.snippets[name]))
        self.assertNotIn("limit", definitions(self.snippets["g"]))
        self.assertIn("limit", declarations(self.snippets["g"]))
        self.assertIn("ext", declarations(self.snippets["g"]))

    def test_only_referenced_metadata_is_carried(self):
        self.assertIn('!"note"', self.snippets["f"])
        self.assertNotIn("never referenced", self.snippets["f"])
        self.assertNotIn('!"note"', self.snippets["g"])

    def test_snippets_are_valid_modules(self):
        for name, snippet in self.snippets.items():
            path = self.directory / f"{name}.ll"
            path.write_text(snippet, encoding="utf-8")
            result = subprocess.run(
                [str(self.tools.llvm_as), str(path), "-o", os.devnull],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_module_state_lists_the_writable_definitions(self):
        self.assertEqual(
            patch_dependencies.writable_definitions(self.module, self.tools),
            ["counter", "limit"],
        )


class UnresolvedSymbols(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = find_tools()
        cls.directory = Path(tempfile.mkdtemp(prefix="patch-dependencies-"))
        source = cls.directory / "needs_helper.ll"
        source.write_text(NEEDS_HELPER, encoding="utf-8")
        cls.object = cls.directory / "needs_helper.o"
        subprocess.run(
            [str(cls.tools.clang), "-target", "arm-linux-gnueabihf", "-mthumb", "-fno-pic",
             "-O0", "-c", str(source), "-o", str(cls.object)],
            check=True, capture_output=True,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory, ignore_errors=True)

    def test_undefined_symbols_of_an_object(self):
        # The unwind index also names __aeabi_unwind_cpp_pr0, through a
        # relocation that changes no bytes; it is not a dependency.
        self.assertEqual(
            patch_dependencies.undefined_symbols(self.object),
            ["__aeabi_uldivmod", "uart_write_line"],
        )

    def test_symbols_defined_in_the_object_are_not_listed(self):
        undefined = patch_dependencies.undefined_symbols(self.object)
        self.assertNotIn("divide", undefined)
        self.assertNotIn("local_helper", undefined)

    def test_unresolved_is_what_nothing_provides(self):
        self.assertEqual(
            patch_dependencies.unresolved(self.object, {"uart_write_line": 0x1CC}),
            ["__aeabi_uldivmod"],
        )
        self.assertEqual(
            patch_dependencies.unresolved(
                self.object, {"uart_write_line": 0x1CC, "__aeabi_uldivmod": 0x400}
            ),
            [],
        )

    def test_writable_global_of_the_module_is_unresolved(self):
        # The snippet only declares the counter, so unless something places
        # it, the inserted code has nowhere to keep it.
        module = self.directory / "module.ll"
        module.write_text(MODULE, encoding="utf-8")
        snippet = self.directory / "f.ll"
        snippet.write_text(
            patch_dependencies.split_module(module, self.tools)["f"], encoding="utf-8"
        )
        obj = self.directory / "f.o"
        subprocess.run(
            [str(self.tools.clang), "-target", "arm-linux-gnueabihf", "-mthumb", "-fno-pic",
             "-O0", "-c", str(snippet), "-o", str(obj)],
            check=True, capture_output=True,
        )
        self.assertEqual(
            patch_dependencies.unresolved(obj, {"uart_write_line": 0x1CC, "g": 0x1001}),
            ["counter"],
        )



def report_of(references: dict[str, list[str]]) -> dict:
    return {
        "functions": [
            {"function": name, "undefined": undefined, "unresolved": []}
            for name, undefined in references.items()
        ]
    }


class InsertionOrder(unittest.TestCase):
    """Functions are linked one at a time against what is already placed."""

    def test_callee_is_placed_before_its_caller(self):
        report = report_of(
            {
                "target": ["patch"],
                "patch": ["helper", "uart_write_line"],
                "helper": ["leaf"],
                "leaf": [],
            }
        )
        order, stuck = patch_dependencies.insertion_order(report, "target")
        self.assertEqual(order, ["leaf", "helper", "patch"])
        self.assertEqual(stuck, [])

    def test_target_is_not_in_the_order(self):
        # It replaces a function of the image and goes last; a reference to it
        # resolves to the address it already has.
        report = report_of({"target": ["patch"], "patch": ["target"]})
        order, stuck = patch_dependencies.insertion_order(report, "target")
        self.assertEqual(order, ["patch"])
        self.assertEqual(stuck, [])

    def test_independent_functions_keep_name_order(self):
        report = report_of({"target": ["b", "a"], "b": [], "a": []})
        self.assertEqual(patch_dependencies.insertion_order(report, "target"), (["a", "b"], []))

    def test_mutual_references_cannot_be_ordered(self):
        report = report_of({"target": ["even"], "even": ["odd"], "odd": ["even"], "leaf": []})
        order, stuck = patch_dependencies.insertion_order(report, "target")
        self.assertEqual(order, ["leaf"])
        self.assertEqual(stuck, ["even", "odd"])

    def test_function_that_calls_itself_is_fine(self):
        report = report_of({"target": ["loop"], "loop": []})
        self.assertEqual(patch_dependencies.insertion_order(report, "target"), (["loop"], []))


if __name__ == "__main__":
    unittest.main()
