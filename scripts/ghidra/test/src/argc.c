// CHECK: {{...}}
int argc(int argc, char **argv) {
    return argc;
}
int main(int a, char **argv)
{
    return argc(a, argv);
}
