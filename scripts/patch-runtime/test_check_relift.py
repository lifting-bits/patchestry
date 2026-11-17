#!/usr/bin/env python3
#
# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#
"""Tests of the parts of check_relift.py that need neither Ghidra nor QEMU."""

import unittest

import check_relift


def body(*calls):
    return {"basic_blocks": {"b0": {"operations": [{"target": key} for key in calls]}},
            "entry_block": "b0"}


def lifted():
    return {"functions": {
        "ram:000002d4": {"name": "target", **body("ram:00001000", "ram:000002cc")},
        "ram:00001000": {"name": "FUN_00001000", **body("ram:00001040", "ram:000002f8")},
        "ram:00001040": {"name": "FUN_00001040", **body()},
        "ram:000002cc": {"name": "irq_off", **body()},
        "ram:000002f8": {"name": "reject", **body()},
        "ram:00000078": {"name": "FUN_00000078", **body()},
    }}


class Select(unittest.TestCase):
    def test_keeps_the_target_and_unnamed_functions_it_reaches(self):
        image = lifted()
        chosen, names = check_relift.select(image, check_relift.locate(image, "target", None))
        self.assertEqual(sorted(names), ["FUN_00001000", "FUN_00001040", "target"])
        bodies = {row["name"] for row in chosen["functions"].values() if "basic_blocks" in row}
        self.assertEqual(bodies, {"target", "FUN_00001000", "FUN_00001040"})
        # A named callee and an unnamed function nobody reaches keep a declaration.
        self.assertEqual(chosen["functions"]["ram:000002cc"], {"name": "irq_off"})
        self.assertEqual(chosen["functions"]["ram:00000078"], {"name": "FUN_00000078"})

    def test_target_is_found_by_name_first(self):
        self.assertEqual(check_relift.locate(lifted(), "target", 0x1000), "ram:000002d4")

    def test_moved_body_is_found_by_address_and_renamed(self):
        image = lifted()
        key = check_relift.locate(image, "FUN_000002d4", 0x1000)
        self.assertEqual(key, "ram:00001000")
        self.assertEqual(image["functions"][key]["name"], "FUN_000002d4")

    def test_missing_target_is_refused(self):
        with self.assertRaises(ValueError):
            check_relift.locate(lifted(), "absent", None)
        with self.assertRaises(ValueError):
            check_relift.locate(lifted(), "absent", 0x9999)

    def test_declaration_is_not_a_target(self):
        image = lifted()
        del image["functions"]["ram:000002d4"]["basic_blocks"]
        with self.assertRaises(ValueError):
            check_relift.locate(image, "target", None)


if __name__ == "__main__":
    unittest.main()
