// RUN: true
extern void halt();

// CWE-22: Injected before mkdir — rejects path traversal and absolute paths.
// Blocks:
//   - Relative traversal: "../" sequences anywhere in the path
//   - Absolute paths: paths starting with "/" that escape the intended directory
void patch__before__mkdir(const void *path) {
    const char *p = (const char *)path;
    if (!p) { halt(); return; }

    // Reject absolute paths — caller should only create subdirectories
    if (p[0] == '/') {
        halt();
        return;
    }

    // Reject relative traversal sequences
    while (*p) {
        if (p[0] == '.' && p[1] == '.' && (p[2] == '/' || p[2] == '\0')) {
            halt();
            return;
        }
        p++;
    }
}
