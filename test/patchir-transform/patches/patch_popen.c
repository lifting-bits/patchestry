// RUN: true
extern void halt();

int str_equal(const char *a, const char *b) {
    while (*a && *b) {
        if (*a != *b) return 0;
        a++; b++;
    }
    return *a == *b;
}

// CWE-94: Injected before popen — validates command against allowlist.
// The vulnerable pattern is popen() called with unsanitized diagnostic
// command string that allows arbitrary code execution.
void patch__before__popen(const void *command) {
    const char *cmd = (const char *)command;
    if (!cmd) { halt(); return; }

    if (str_equal(cmd, "self_test") ||
        str_equal(cmd, "sensor_check") ||
        str_equal(cmd, "flow_calibration")) {
        return;
    }
    halt();
}
