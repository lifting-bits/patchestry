// UNSUPPORTED: system-windows
// RUN: %cc-x86_64 %s -g -c -o %t.o
// RUN: %decompile-headless --input %t.o --function leaf --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=SCHEMA %s --input-file %t

// Schema enrichment landed alongside TailCallAnalysis (bepdg-gen issue
// #99). Every function gets a first-class entry_point and an
// address_ranges array; the JSON tail carries a boundary_repairs
// array that is empty when the analyzer found nothing to flag.
//
// SCHEMA: "name":"{{_?leaf}}"
// SCHEMA-SAME: "entry_point":"
// SCHEMA-SAME: "address_ranges":[{"start":"
// SCHEMA: "boundary_repairs":[]

int leaf(int a, int b) { return a + b; }
