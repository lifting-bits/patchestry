// UNSUPPORTED: system-windows
// RUN: %cc %s -o %t
// RUN %t; if [ "$(uname)" = "Linux" ]; then %decompile-headless %t main %t1 fi
// RUN %t; if [ "$(uname)" = "Darwin" ]; then %decompile-headless %t _main %t1 fi
// RUN %t1; %file-check %s --input-file %t1
// CHECK: {{...}}

struct data
{
    int a;
    int b;
    int c;
    int d;
    int e;
};

int main(int argc, char **argv)
{
    struct data d = { 0, 1, 2, 3, 4 };
    return d.a + d.b + d.c + d.d + d.e;
}
