// UNSUPPORTED: system-windows
// Ghidra 12.2 can report a bit-field read as ZPULL (unsigned) or SPULL
// (signed) and a bit-field write as INSERT.  That rewrite follows the source,
// not the machine code: reg_update below reads the register once and writes it
// once, and the rewrite turns it into two read-modify-write sequences.  The
// export scripts turn the rewrite off, so the export has the shifts and masks
// of the binary and each function has the loads and stores the binary has.
// test/patchir-decomp/bitfield_shift_mask.json is the export of this file.
// RUN: %cc-aarch64 %s -g -O1 -c -o %t.o
// RUN: %decompile-headless --input %t.o --output %t %ci_output_folder
// RUN: %file-check -vv --check-prefix=DECOMPILEA %s --input-file %t
// DECOMPILEA-DAG: "name":"bitfield_get_unsigned"
// DECOMPILEA-DAG: "name":"bitfield_set"
// DECOMPILEA-DAG: "name":"reg_update"
// RUN: %file-check -vv --check-prefix=NOPULL %s --input-file %t
// NOPULL-NOT: "mnemonic":"ZPULL"
// NOPULL-NOT: "mnemonic":"SPULL"
// NOPULL-NOT: "mnemonic":"INSERT"

// One read of the register and one write to it.
// RUN: %decompile-headless --input %t.o --function reg_update --output %t.reg %ci_output_folder
// RUN: %file-check -vv --check-prefix=LOADS %s --input-file %t.reg
// LOADS: "mnemonic":"LOAD"
// LOADS-NOT: "mnemonic":"LOAD"
// RUN: %file-check -vv --check-prefix=STORES %s --input-file %t.reg
// STORES: "mnemonic":"STORE"
// STORES-NOT: "mnemonic":"STORE"

struct flags {
    unsigned int mode : 3;
    int delta : 5;
    unsigned int count : 12;
    int bias : 12;
};

struct wide {
    unsigned long tag : 7;
    long offset : 40;
    unsigned long rest : 17;
};

struct small {
    unsigned char low : 2;
    signed char mid : 3;
    unsigned char high : 3;
};

// The bit-field unit is 4 bytes of a 48-byte struct.
struct big {
    unsigned int mode : 3;
    int delta : 5;
    unsigned int count : 12;
    int bias : 12;
    long tail[4];
    int last;
};

// The bit-field unit starts at byte 1.
struct __attribute__((packed)) packed {
    unsigned char tag;
    unsigned int lo : 6;
    unsigned int mid : 20;
    unsigned int hi : 6;
    unsigned char end;
};

struct reg {
    unsigned int enable : 1;
    unsigned int mode : 3;
    unsigned int divider : 12;
    unsigned int reserved : 16;
};

unsigned int bitfield_get_unsigned(const struct flags *f) { return f->count; }

int bitfield_get_signed(const struct flags *f) { return f->delta; }

int bitfield_get_signed_top(const struct flags *f) { return f->bias; }

void bitfield_set(struct flags *f, unsigned int count, int delta) {
    f->count = count;
    f->delta = delta;
}

long bitfield_wide_get(const struct wide *w) { return w->offset; }

void bitfield_wide_set(struct wide *w, long offset) { w->offset = offset; }

int bitfield_small_get(const struct small *s) { return s->mid; }

void bitfield_small_set(struct small *s, int mid, unsigned high) {
    s->mid = mid;
    s->high = high;
}

void big_set(struct big *b, unsigned int count) { b->count = count; }

void packed_set(struct packed *p, unsigned int mid) { p->mid = mid; }

// Reads the register once, updates two fields in a copy, writes it once.
void reg_update(volatile struct reg *r, unsigned int mode, unsigned int divider) {
    struct reg v = *r;
    v.mode = mode;
    v.divider = divider;
    *r = v;
}
