// UNSUPPORTED: system-windows
// RUN: %cc-x86_64 %s -g -c -o %t.o
// RUN: %decompile-headless --input %t.o --function widechar_test --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILES %s --input-file %t
// DECOMPILES: "name":"{{_?widechar_test}}"

// RUN: %decompile-headless --input %t.o --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILEA %s --input-file %t
// DECOMPILEA: "architecture":"{{.*}}","format":"{{.*}}","functions":
// DECOMPILEA-SAME: "name":"{{_?widechar_test}}"
// DECOMPILEA-SAME: "name":"{{_?main}}"

// RUN: %decompile-headless --input %t.o --list-functions --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=LISTFNS %s --input-file %t
// LISTFNS: "program":"{{.*}}","functions":
// LISTFNS-SAME: "name":"{{_?widechar_test}}"
// LISTFNS-SAME: "name":"{{_?main}}"

#include <stdio.h>
#include <wchar.h>
#include <locale.h>

int widechar_test() {
    // Set the locale to the user's preferred locale
    setlocale(LC_ALL, "");

    // Declare and initialize a wide string (wide character array)
    wchar_t wide_str[] = L"Hello, 世界!"; // "Hello, World!" in Chinese
    // Print the wide string
    wprintf(L"%ls\n", wide_str);
    // Print a wide character
    wchar_t wide_char = L'😊'; // A smiley emoji
    wprintf(L"Wide character: %lc\n", wide_char);
    return 0;
}

int main(void) {
    return widechar_test();
}