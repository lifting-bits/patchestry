// UNSUPPORTED: system-windows
// RUN: %cc-x86_64 %s -g -c -o %t.o
// RUN: %decompile-headless --input %t.o --function union_test --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILES %s --input-file %t
// DECOMPILES: "name":"{{_?union_test}}"

// RUN: %decompile-headless --input %t.o --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILEA %s --input-file %t
// DECOMPILEA: "architecture":"{{.*}}","format":"{{.*}}","functions":
// DECOMPILEA-SAME: "name":"{{_?union_test}}"
// DECOMPILEA-SAME: "name":"{{_?main}}"

// RUN: %decompile-headless --input %t.o --list-functions --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=LISTFNS %s --input-file %t
// LISTFNS: "program":"{{.*}}","functions":
// LISTFNS-SAME: "name":"{{_?union_test}}"
// LISTFNS-SAME: "name":"{{_?main}}"

struct access
{
    int l;
    int h;
};

union data {
    unsigned long long b;
    struct access s;
};

int union_test(int argc, char **argv) {
    union data d;
    d.b = 0xffffffff00000000 + argc;
    return d.s.l;
}

int main(int argc, char **argv) { return union_test(argc, argv); }
