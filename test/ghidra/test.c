// UNSUPPORTED: system-windows
// RUN: %cc %s -o %t && %decompile-headless %t main %t1 && %file-check %s --input-file %t1
// CHECK: {{...}}

#include <stdio.h>

int main() {
  printf("Test passed\n");
  return 0;
}
