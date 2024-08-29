// UNSUPPORTED: system-windows
// RUN: %cc %s -o %t && %decompile-headless %t sub %t1 && %file-check %s --input-file %t1
// CHECK: {{...}}

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
