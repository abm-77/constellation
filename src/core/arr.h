#include <core/language_layer.h>
#include <core/mem.h>

#if __STDC_VERSION >= 202311L
#define core_typeof(T) typeof(T)
#elif defined(__GNUC__) 
#define core_typeof(T) __typeof__(T)
#endif

typedef struct CoreStackVoid {
    void* data;
    u32 len;
    u32 cap;
} CoreStackVoid;

#define CORE_STACK_DEFINE(T)                            \
    typedef union CoreStack_##T {                       \
        CoreStackVoid void_stack;                       \
        struct {                                        \
            T* data;                                    \
            u32 len;                                    \
            u32 cap;                                    \
        };                                              \
    } CoreStack_##T                                     \

#define CoreStack(T) CoreStack_##T

void core_stack_init_void(CoreStackVoid* stack, Arena* arena, u32 cap, u32 elt_size, u32 elt_align) {
}

#define core_stack_init(stack, arena, cap) core_stack_init_void(&stack->void_stack, arena, sizeof(*stack->data), alignof(core_typeof(stack.data)))


CORE_STACK_DEFINE(i32);
CORE_STACK_DEFINE(i64);
CORE_STACK_DEFINE(u8);
CORE_STACK_DEFINE(u32);
CORE_STACK_DEFINE(u64);
CORE_STACK_DEFINE(f32);
CORE_STACK_DEFINE(f64);

CORE_STACK_DEFINE(char);

typedef char* CHAR_PTR;
CORE_STACK_DEFINE(CHAR_PTR);

