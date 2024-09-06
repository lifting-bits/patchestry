// UNSUPPORTED: system-windows
// RUN: %cc %s -g -o %t.o
// RUN: %decompile-headless --input %t.o --function sort_test --output %t
// RUN: %file-check -vv %s --input-file %t
// CHECK: "name":"{{_?sort_test}}"

#include <stdio.h>

int sort_test()
{
  int array[100], n, c, d, swap;
  scanf("%d", &n);
  for (c = 0; c < n; c++)
    scanf("%d", &array[c]);

  for (c = 0 ; c < n - 1; c++) {
    for (d = 0 ; d < n - c - 1; d++) {
      if (array[d] > array[d+1]) {
        swap       = array[d];
        array[d]   = array[d+1];
        array[d+1] = swap;
      }
    }
  }

  for (c = 0; c < n; c++)
     printf("%d\n", array[c]);

  return 0;
}

int main(void) {
  return sort_test();
}
