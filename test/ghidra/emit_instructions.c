// --emit-instructions plumbing, end-to-end through decompile-headless.sh.
// Off by default. When on, every function carries an `instructions` map
// keyed by address, emitted right after address_ranges, with the
// disassembly text, the encoded length and the raw P-Code of each
// instruction.
//
// UNSUPPORTED: system-windows
// RUN: %cc-x86_64 %s -g -c -o %t.o

// Default: no instructions key anywhere in the document.
// RUN: %decompile-headless --input %t.o --function leaf --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=OFF %s --input-file %t
// OFF: "name":"{{_?leaf}}"
// OFF-NOT: "instructions"

// RUN: %decompile-headless --input %t.o --function leaf --output %t --emit-instructions %ci_output_folder
// RUN: %file-check -vv --check-prefix=ON %s --input-file %t
// ON: "name":"{{_?leaf}}"
// ON: "address_ranges":[{"start":"
// ON: "instructions":{"ram:{{[0-9a-f]+}}":{"text":"{{[^"]+}}","length":{{[0-9]+}},"pcode":[

// The =off spelling goes through the same forwarding path.
// RUN: %decompile-headless --input %t.o --function leaf --output %t --emit-instructions=off %ci_output_folder
// RUN: %file-check -vv --check-prefix=OFF %s --input-file %t

int leaf(int a, int b) { return a + b; }
