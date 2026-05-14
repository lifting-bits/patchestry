// UNSUPPORTED: system-windows
// RUN: %cc-x86_64 %s -g -c -o %t.o
// RUN: %decompile-headless --input %t.o --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=CONTRACT %s --input-file %t
//
// #226 contract: a function's address never appears in globals[] under
// the same name.  Take the address of a local function and confirm the
// name appears in functions[] only.  Functions: comes before globals:
// in the JSON layout, so CHECK-NOT after the SAME match catches any
// globals[] alias.
//
// CONTRACT:           "functions":
// CONTRACT-SAME:      "name":"{{_?leaf_226}}"
// CONTRACT-NOT:       "name":"{{_?leaf_226}}"

static void leaf_226(int x) {
    (void)x;
}

typedef void (*fp_t)(int);

fp_t take_addr_of_leaf_226(void) {
    return &leaf_226;
}

int main(int argc, char **argv) {
    (void)argv;
    fp_t fp = take_addr_of_leaf_226();
    fp(argc);
    return 0;
}
