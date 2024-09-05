// UNSUPPORTED: system-windows
// RUN: %cc %s -o %t.o
// RUN: %decompile-headless --input %t.o --function argc --output %t
// RUN: %file-check -vv %s --input-file %t
// CHECK: "name":"{{_?argc}}"

int argc(int argc, char **argv) { return argc; }

int main(int a, char **argv)
{
    return argc(a, argv);
}
