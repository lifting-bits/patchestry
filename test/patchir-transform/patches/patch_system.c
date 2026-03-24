// RUN: true
typedef unsigned long size_t;
typedef unsigned int uint32_t;

extern unsigned int system(const void *command);

// CWE-78: Replace system() with a version that rejects shell metacharacters.
// The vulnerable pattern is system() called with unsanitized user input
// concatenated into a command string.
// Note: void* param and unsigned return match CIR types from decompiled binary.
unsigned int patch__replace__system(const void *command, uint32_t max_len) {
    const char *cmd = (const char *)command;
    if (!cmd) return (unsigned int)-1;

    const char *reject = ";|&`$(){}[]<>!\\\"'*?\n\r";
    for (uint32_t i = 0; i < max_len && cmd[i] != '\0'; i++) {
        for (const char *r = reject; *r; r++) {
            if (cmd[i] == *r) return (unsigned int)-1;
        }
    }
    return system(command);
}
