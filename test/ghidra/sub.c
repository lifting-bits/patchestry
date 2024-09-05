// UNSUPPORTED: system-windows
// RUN: %cc %s -g -o %t.o
// RUN: %decompile-headless --input %t.o --function sub --output %t %ci_output_folder
// RUN: %file-check -vv %s --input-file %t
// CHECK: "name":"{{_?sub}}"

#include <stdio.h>

int sub()
{
	int x;
	x = 4;
	return x - 4;
}

int main(void) {
	printf("sub: %d\n", sub());
	return 0;
}
