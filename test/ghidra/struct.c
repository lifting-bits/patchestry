// UNSUPPORTED: system-windows
// RUN: %cc %s -g -o %t.o
// RUN: %decompile-headless --input %t.o --function struct_test --output %t %ci_output_folder
// RUN: %file-check -vv %s --input-file %t
// CHECK: "name":"{{_?struct_test}}"

struct data
{
    int a;
    int b;
    int c;
    int d;
    int e;
};

int struct_test(int argc, char **argv)
{
    struct data d = { 0, 1, 2, 3, 4 };
    return d.a + d.b + d.c + d.d + d.e;
}

int main(int argc, char **argv) {
    return struct_test(argc, argv);

}
