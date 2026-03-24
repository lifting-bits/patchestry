// RUN: true
typedef unsigned int uint32_t;
typedef unsigned char bool_t;

// CWE-193: Replace off-by-one bounds check.
// Original uses >= (allows index == size, which is one past the end).
// Fixed version uses > (rejects index == size).
bool_t patch__replace__bounds_cmp(unsigned int index, unsigned int size) {
    if (index > size) {
        return 1;
    }
    return 0;
}
