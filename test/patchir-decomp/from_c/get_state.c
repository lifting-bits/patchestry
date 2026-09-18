// patchestry:tu format=1 target=ARM:LE:32:Cortex arch=ARM
struct device_ctx;
typedef struct device_ctx device_ctx_t;
struct device_ctx {
    int fd;
    int flags;
};
extern int g_state;
int get_state(int count);
// patchestry:function-begin ram:20000000 name=get_state symbol=get_state
int get_state(int count) {
    return count + g_state;
}
// patchestry:function-end ram:20000000
