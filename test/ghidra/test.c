// UNSUPPORTED: system-windows
// RUN: %cc %s -o %t && %decompile-headless %t test %t1 && %file-check %s --input-file %t1
// CHECK: {{...}}

#include <stdio.h>

int test() {
  printf("Test passed\n");
  return 0;
}

int main(void) {
  return test();
}