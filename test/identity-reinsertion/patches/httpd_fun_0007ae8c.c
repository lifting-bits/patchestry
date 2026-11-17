/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

/*
 * Patched versions of `httpd` FUN_0007ae8c, written from its lifted C.
 *
 * The function copies the path token of a request line (param_2) into
 * param_1 and has no bound on the copy.  With LIMIT defined, a token of LIMIT
 * bytes or more is rejected by returning a null pointer; the original never
 * returns one.  That check sits in the copy loop, so a rejected token has
 * already been copied in part.  With LIMIT_FIRST defined as well, the token
 * is measured before anything is written.  RETURN_OFFSET, when defined, is added to the returned pointer:
 * a change no input-restricting patch may make.  So are EXTRA_WRITE, which
 * stores a byte at that offset of the output buffer, and EXTRA_CALL, which
 * calls an external function the original does not call.
 *
 * Without either macro this is the original, with the dereferences written
 * as the lifted IR has them.
 */

typedef unsigned char undefined1;

#ifdef LIMIT
#define CHECK_LENGTH(count) if ((count) >= LIMIT) return (undefined1 *)0
#else
#define CHECK_LENGTH(count)
#endif

#ifndef RETURN_OFFSET
#define RETURN_OFFSET 0
#endif

#define BYTE_AT(address) (*(signed char *)(address))

#ifdef EXTRA_CALL
extern void audit_request(int);
#endif

undefined1 *FUN_0007ae8c(undefined1 *param_1, int param_2, int param_3) {
    int local_10;
    int local_c;
    local_10 = 0;
    local_c = 0;
#if defined(LIMIT) && defined(LIMIT_FIRST)
    {
        /* The length of the token the loops below copy. */
        int at = 0;
        int count = 0;
        while (BYTE_AT(param_2 + at) == 32)
            at = at + 1;
        while (BYTE_AT(param_2 + at) != 32 && BYTE_AT(param_2 + at) != 0)
            at = at + 1;
        while (BYTE_AT(param_2 + at) == 32)
            at = at + 1;
        while (BYTE_AT(param_2 + at) != 32 && BYTE_AT(param_2 + at) != 0) {
            at = at + 1;
            count = count + 1;
        }
        if (count > LIMIT)
            return (undefined1 *)0;
    }
#endif
#ifdef EXTRA_WRITE
    param_1[EXTRA_WRITE] = 1U;
#endif
#ifdef EXTRA_CALL
    audit_request(param_3);
#endif
    while (BYTE_AT(param_2 + local_c) == 32)
        local_c = local_c + 1;
    while (BYTE_AT(param_2 + local_c) != 32) {
        if (BYTE_AT(param_2 + local_c) == 0)
            break;
        local_c = local_c + 1;
    }
    if (BYTE_AT(param_2 + local_c) == 0) {
        *param_1 = 47U;
        param_1[1] = 0U;
    } else {
        while (BYTE_AT(param_2 + local_c) == 32)
            local_c = local_c + 1;
        if (BYTE_AT(param_2 + local_c) == 47) {
            while (BYTE_AT(param_2 + local_c) != 32) {
                if (BYTE_AT(param_2 + local_c) == 0)
                    break;
                CHECK_LENGTH(local_10);
                param_1[local_10] = *(undefined1 *)(param_2 + local_c);
                local_10 = local_10 + 1;
                local_c = local_c + 1;
            }
            param_1[local_10] = 47U;
            param_1[local_10 + 1] = 0U;
        } else if (BYTE_AT(param_2 + local_c) != 47) {
            while (BYTE_AT(param_2 + local_c) != 32) {
                if (BYTE_AT(param_2 + local_c) == 0)
                    break;
                CHECK_LENGTH(local_10);
                param_1[param_3 + local_10] = *(undefined1 *)(param_2 + local_c);
                local_10 = local_10 + 1;
                local_c = local_c + 1;
            }
            param_1[param_3 + local_10] = 47U;
            param_1[param_3 + local_10 + 1] = 0U;
        }
    }
    return param_1 + RETURN_OFFSET;
}
