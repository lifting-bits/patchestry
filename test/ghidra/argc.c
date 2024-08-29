// UNSUPPORTED: system-windows
// RUN: %cc %s -o %t && %decompile-headless %t argc %t1 && %file-check -vv %s --input-file %t1
// CHECK: {{...}}

int argc(int argc, char **argv) {
    return argc;
}
int main(int a, char **argv)
{
    return argc(a, argv);
}
