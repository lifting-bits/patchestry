// RUN: true
typedef unsigned int uint32_t;
typedef unsigned char bool_t;

// CWE-193: Replace off-by-one wrap condition in circular buffer Put().
// The vulnerable code uses >= N to wrap, which skips index N entirely.
// The buffer is sized N+1, so valid indices are 0 through N.
// The fix uses > N so that index N is reachable before wrapping.
bool_t patch__replace__wrap_cmp(unsigned int new_head, unsigned int capacity) {
    if (new_head > capacity) {
        return 1;
    }
    return 0;
}
