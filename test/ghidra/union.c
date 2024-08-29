// UNSUPPORTED: system-windows
// RUN: %cc %s -o %t && %decompile-headless %t main %t1 && %file-check %s --input-file %t1
// CHECK: {{...}}

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

int main(int argc, char **argv)
{
    union data d;
    d.b = 0xffffffff00000000 + argc;
    return d.s.l;
}
