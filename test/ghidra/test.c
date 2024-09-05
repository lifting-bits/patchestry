// UNSUPPORTED: system-windows
// RUN: %cc %s -o %t.o
// RUN: %decompile-headless --input %t.o --function test --output %t
// RUN: %file-check -vv %s --input-file %t
// CHECK: "name":"{{_?test}}"

#include <stdio.h>

int test() {
  printf("Test passed\n");
  return 0;
}

int main(void) {
  return test();
}
