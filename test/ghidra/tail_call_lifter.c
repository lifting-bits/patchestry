// UNSUPPORTED: system-windows
// RUN: %cc-x86_64 %s -g -O2 -c -o %t.o
// RUN: %decompile-headless --input %t.o --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=LIFTER %s --input-file %t

// Verifies that TailCallAnalysis lifts a clang::musttail BRANCH to the
// TAIL_CALL CALLOTHER op (mnemonic "TAIL_CALL"). Detection-only mode is
// the default — no --repair-function-boundaries flag is needed; the
// inner function is already its own Ghidra function (it's externally
// visible via the call from main), so we're testing the lifter half of
// the pipeline (BN's translateTailCalls analogue), not the splitter.
//
// LIFTER: "mnemonic":"TAIL_CALL"

extern int sink(int x);

int wrapper(int x) {
    [[clang::musttail]] return sink(x);
}

int main(int argc, char **argv) {
    return wrapper(argc);
}
