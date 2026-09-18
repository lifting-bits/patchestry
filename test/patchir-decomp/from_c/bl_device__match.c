// patchestry:tu format=1 target=ARM:LE:32:v7 arch=ARM
struct _IO_FILE;
struct _IO_marker;
typedef unsigned int undefined4;
typedef unsigned int size_t;
typedef unsigned char undefined;
typedef void _IO_lock_t;
typedef long long __off64_t;
typedef struct _IO_FILE FILE;
typedef unsigned char undefined1;
typedef int __off_t;
struct _IO_marker {
    struct _IO_marker *_next;
    struct _IO_FILE *_sbuf;
    int _pos;
};
struct _IO_FILE {
    int _flags;
    signed char *_IO_read_ptr;
    signed char *_IO_read_end;
    signed char *_IO_read_base;
    signed char *_IO_write_base;
    signed char *_IO_write_ptr;
    signed char *_IO_write_end;
    signed char *_IO_buf_base;
    signed char *_IO_buf_end;
    signed char *_IO_save_base;
    signed char *_IO_backup_base;
    signed char *_IO_save_end;
    struct _IO_marker *_markers;
    struct _IO_FILE *_chain;
    int _fileno;
    int _flags2;
    __off_t _old_offset;
    unsigned short _cur_column;
    signed char _vtable_offset;
    signed char _shortbuf[1];
    _IO_lock_t *_lock;
    __off64_t _offset;
    void *__pad1;
    void *__pad2;
    void *__pad3;
    void *__pad4;
    size_t __pad5;
    int _mode;
    signed char _unused2[40];
};
extern FILE *stderr;
extern undefined DAT_00028e20;
int udev_device_new_from_syspath(int param_0, int param_1);
int udev_enumerate_scan_devices(int param_0);
int udev_device_get_parent_with_subsystem_devtype(int param_0, undefined *param_1, signed char *param_2);
size_t fwrite(void *param_0, size_t param_1, size_t param_2, FILE *param_3);
int udev_enumerate_new(int param_0);
void udev_device_unref(int param_0);
signed char *strdup(signed char *param_0);
undefined4 udev_device_get_sysattr_value(int param_0, signed char *param_1);
undefined4 udev_enumerate_get_list_entry(int param_0);
int udev_list_entry_get_name(undefined4 param_0);
void udev_unref(int param_0);
undefined1 bl_device__match(undefined4 param_1, undefined4 *param_2);
int udev_new(void);
void udev_enumerate_unref(int param_0);
int puts(signed char *param_0);
int strcmp(signed char *param_0, signed char *param_1);
int udev_enumerate_add_match_sysname(int param_0, undefined4 param_1);
// patchestry:function-begin ram:00022cd4 name=bl_device__match symbol=bl_device__match
undefined1 bl_device__match(undefined4 param_1, undefined4 *param_2) {
    undefined1 local_35;
    int iVar1;
    int iVar2;
    int iVar3;
    undefined4 uVar4;
    int iVar5;
    signed char *pcVar6;
    signed char *__s1;
    int iVar7;
    undefined4 __call_ret_0;
    undefined4 __call_ret_1;
    undefined4 __call_ret_2;
    size_t __call_ret_3;
    size_t __call_ret_4;
    int __call_ret_5;
    size_t __call_ret_6;
    size_t __call_ret_7;
    size_t __call_ret_8;
    size_t __call_ret_9;
    local_35 = 0U;
    iVar1 = udev_new();
    if (iVar1 != 0) {
        iVar2 = udev_enumerate_new(iVar1);
        if (iVar2 == 0) {
            __call_ret_8 = fwrite((void *)"Can't create udev enumerate context.\n", 1U, 37U, stderr);
            udev_unref(iVar1);
            return local_35;
        }
        iVar3 = udev_enumerate_add_match_sysname(iVar2, param_1);
        if (iVar3 >= 0) {
            iVar3 = udev_enumerate_scan_devices(iVar2);
            if (iVar3 < 0) {
                __call_ret_6 = fwrite((void *)"Failed on scanning device.\n", 1U, 27U, stderr);
                udev_enumerate_unref(iVar2);
                udev_unref(iVar1);
                return local_35;
            }
            uVar4 = udev_enumerate_get_list_entry(iVar2);
            iVar3 = udev_list_entry_get_name(uVar4);
            if (iVar3 == 0) {
                __call_ret_5 = puts((signed char *)"No matched device found.");
                udev_enumerate_unref(iVar2);
                udev_unref(iVar1);
                return local_35;
            }
            iVar3 = udev_device_new_from_syspath(iVar1, iVar3);
            if (iVar3 == 0) {
                __call_ret_4 = fwrite((void *)"Unable to find usb device.", 1U, 26U, stderr);
                udev_enumerate_unref(iVar2);
                udev_unref(iVar1);
                return local_35;
            }
            iVar5 = udev_device_get_parent_with_subsystem_devtype(iVar3, &DAT_00028e20, (signed char *)"usb_device");
            if (iVar5 == 0) {
                __call_ret_3 = fwrite((void *)"Unable to find parent usb device.", 1U, 33U, stderr);
                udev_device_unref(iVar3);
                udev_enumerate_unref(iVar2);
                udev_unref(iVar1);
                return local_35;
            }
            __call_ret_0 = udev_device_get_sysattr_value(iVar5, (signed char *)"manufacturer");
            pcVar6 = (signed char *)__call_ret_0;
            __call_ret_1 = udev_device_get_sysattr_value(iVar5, (signed char *)"product");
            __s1 = (signed char *)__call_ret_1;
            if (pcVar6 == (signed char *)(void *)0U) {
                udev_device_unref(iVar3);
                udev_enumerate_unref(iVar2);
                udev_unref(iVar1);
                return local_35;
            }
            if (__s1 == (signed char *)(void *)0U) {
                udev_device_unref(iVar3);
                udev_enumerate_unref(iVar2);
                udev_unref(iVar1);
                return local_35;
            }
            iVar7 = strcmp(pcVar6, (signed char *)"Codethink");
            if (iVar7 != 0) {
                udev_device_unref(iVar3);
                udev_enumerate_unref(iVar2);
                udev_unref(iVar1);
                return local_35;
            }
            iVar7 = strcmp(__s1, (signed char *)"Bloodlight");
            if (iVar7 != 0) {
                udev_device_unref(iVar3);
                udev_enumerate_unref(iVar2);
                udev_unref(iVar1);
                return local_35;
            }
            __call_ret_2 = udev_device_get_sysattr_value(iVar5, (signed char *)"serial");
            pcVar6 = (signed char *)__call_ret_2;
            local_35 = 1U;
            pcVar6 = strdup(pcVar6);
            *param_2 = (undefined4)pcVar6;
            udev_device_unref(iVar3);
            udev_enumerate_unref(iVar2);
            udev_unref(iVar1);
            return local_35;
        }
        __call_ret_7 = fwrite((void *)"Failed on adding udev enumerate filter.\n", 1U, 40U, stderr);
        udev_enumerate_unref(iVar2);
        udev_unref(iVar1);
        return local_35;
    }
    __call_ret_9 = fwrite((void *)"Can't create udev\n", 1U, 18U, stderr);
    return local_35;
}
// patchestry:function-end ram:00022cd4
