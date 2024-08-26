// UNSUPPORTED: system-windows
// RUN: %cc %s -o %t
// RUN %t; if [ "$(uname)" = "Linux" ]; then %decompile-headless %t argc %t1 fi
// RUN %t; if [ "$(uname)" = "Darwin" ]; then %decompile-headless %t _argc %t1 fi
// RUN %t1; %file-check %s --input-file %t1
// CHECK: {{...}}

int argc(int argc, char **argv) {
    return argc;
}
int main(int a, char **argv)
{
    return argc(a, argv);
}
