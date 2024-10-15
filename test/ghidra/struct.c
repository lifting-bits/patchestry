// UNSUPPORTED: system-windows
// RUN: %cc %s -g -o %t.o
// RUN: %decompile-headless --input %t.o --function struct_test --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILES %s --input-file %t
// DECOMPILES: "name":"{{_?struct_test}}"

// RUN: %decompile-headless --input %t.o --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILEA %s --input-file %t
// DECOMPILEA: "arch":"{{.*}}","format":"{{.*}}","functions":{{...}}
// DECOMPILEA-SAME: "name":"{{_?struct_test}}"
// DECOMPILEA-SAME: "name":"{{_?main}}"

// RUN: %decompile-headless --input %t.o --list-functions --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=LISTFNS %s --input-file %t
// LISTFNS: "program":"{{.*}}","functions":{{...}}
// LISTFNS-SAME: "name":"{{_?struct_test}}"
// LISTFNS-SAME: "name":"{{_?main}}"

struct data
{
    int a;
    int b;
    int c;
    int d;
    int e;
};

int struct_test(int argc, char **argv) {
    struct data d = { 0, 1, 2, 3, 4 };
    return d.a + d.b + d.c + d.d + d.e;
}

int main(int argc, char **argv) { return struct_test(argc, argv); }
