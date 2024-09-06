// UNSUPPORTED: system-windows
// RUN: %cc %s -g -o %t.o
// RUN: %decompile-headless --input %t.o --function union_test --output %t
// RUN: %file-check -vv %s --input-file %t
// CHECK: "name":"{{_?union_test}}"

struct access
{
    int l;
    int h;
};

union data
{
    unsigned long long b;
    struct access s;
};

int union_test(int argc, char **argv)
{
    union data d;
    d.b = 0xffffffff00000000 + argc;
    return d.s.l;
}

int main(int argc, char **argv) {
    return union_test(argc, argv);
}
