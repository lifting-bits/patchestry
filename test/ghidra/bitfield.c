// UNSUPPORTED: system-windows
// Ghidra 12.2 reports a bit-field read as ZPULL (unsigned) or SPULL (signed)
// and a bit-field write as INSERT, where earlier releases reported shifts and
// masks.  test/patchir-decomp/bitfield_pull_insert.json is the export of this
// file and checks that the lifter handles the three operations.
// RUN: %cc-aarch64 %s -g -O1 -c -o %t.o
// RUN: %decompile-headless --input %t.o --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILEA %s --input-file %t
// DECOMPILEA-DAG: "name":"bitfield_get_unsigned"
// DECOMPILEA-DAG: "name":"bitfield_set"
// DECOMPILEA-DAG: "mnemonic":"ZPULL"
// DECOMPILEA-DAG: "mnemonic":"SPULL"
// DECOMPILEA-DAG: "mnemonic":"INSERT"

struct flags {
    unsigned int mode : 3;
    int delta : 5;
    unsigned int count : 12;
    int bias : 12;
};

struct wide {
    unsigned long long tag : 7;
    long long offset : 40;
    unsigned long long rest : 17;
};

struct small {
    unsigned char low : 2;
    signed char mid : 3;
    unsigned char high : 3;
};

unsigned int bitfield_get_unsigned(const struct flags *f) { return f->count; }

int bitfield_get_signed(const struct flags *f) { return f->delta; }

int bitfield_get_signed_top(const struct flags *f) { return f->bias; }

void bitfield_set(struct flags *f, unsigned int count, int delta) {
    f->count = count;
    f->delta = delta;
}

long long bitfield_wide_get(const struct wide *w) { return w->offset; }

void bitfield_wide_set(struct wide *w, long long offset) { w->offset = offset; }

int bitfield_small_get(const struct small *s) { return s->mid; }

void bitfield_small_set(struct small *s, int mid, unsigned high) {
    s->mid = mid;
    s->high = high;
}
