// RUN: %patchir-c-normalize -input %s -print-tu -output %t >> /dev/null 2>&1
// RUN: %file-check -vv -check-prefix=SWITCH %s --input-file %t.c
// SWITCH: switch (
// SWITCH: case

// Pattern: BRANCHIND (indirect goto via address-of-label table) →
// SwitchRecoveryPass reconstructs a switch/case structure.
// GNU address-of-label extension (&&label) with an indirect goto (goto *ptr).

int switch_dispatch(int sel) {
    int result = 0;
    static void *tbl[] = { &&case0, &&case1, &&case2, &&default_case };
    if ((unsigned)sel >= 4) goto default_case;
    goto *tbl[sel];
case0:
    result = 10;
    goto done;
case1:
    result = 20;
    goto done;
case2:
    result = 30;
    goto done;
default_case:
    result = -1;
done:
    return result;
}
