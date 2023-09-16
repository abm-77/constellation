#pragma once

#include "language_layer.h"

#define arena_push_array(a,T,c) (T*)arena_alloc(a, sizeof(T)*c)
#define arena_push(a,T) arena_push_array(a, T, 1)
#define arena_def(a,T,n,val)\
    T* n = arena_push(a, T);\
    *(n) = (val)\

#define MEM_DEFAULT_ALIGNMENT sizeof(void*)

typedef struct Arena {
	byte* data;
	u32 alloc_pos;
	u32 capacity;
} Arena;

Arena arena_create(u32 size) {
    Arena arena = {
        .data = (byte*) malloc(size),
        .alloc_pos = 0,
        .capacity = size,
    };
    return arena;
}

byte* arena_alloc_align(Arena* arena, u32 size, u32 align) {
    byte* res = NULL;
    u32 alloc_size = AlignUpPow2(size, align);
    if (arena->alloc_pos + alloc_size <= arena->capacity) {
        res = arena->data + arena->alloc_pos;
        MemoryZero(res, alloc_size);
        arena->alloc_pos += alloc_size;
    }
    return res;
}

byte* arena_alloc(Arena* arena, u32 size) {
    return arena_alloc_align(arena, size, MEM_DEFAULT_ALIGNMENT);
}

void arena_dealloc_align(Arena* arena, u32 size, u32 align) {
    u32 dealloc_size = AlignUpPow2(size, align);
    arena->alloc_pos = ClampBot(arena->alloc_pos - dealloc_size, 0);
}

void arena_dealloc_to_align(Arena* arena, u32 pos, u32 align) {
    u32 pos_diff = AlignUpPow2(ClampBot(arena->alloc_pos - pos, 0), align);
    arena->alloc_pos -= pos_diff;
}

void arena_dealloc_to(Arena* arena, u32 pos) {
    arena_dealloc_to_align(arena, pos, MEM_DEFAULT_ALIGNMENT);
}

void arena_dealloc(Arena* arena, u32 size) {
    arena_dealloc_align(arena, size, MEM_DEFAULT_ALIGNMENT);
}

void clear(Arena* arena) {
    arena->alloc_pos = 0;
}

void arena_release(Arena* arena) {
    free(arena->data);
    arena->alloc_pos = 0;
    arena->capacity = 0;
}

typedef struct TempArena {

    Arena* arena;
    u32 start_pos;
} TempArena;

TempArena temp_arena_begin(Arena* arena) {
    TempArena tmp = {
        .arena = arena,
        .start_pos = arena->alloc_pos,
    };
    return tmp;
}
void temp_arena_end(TempArena* tmp) {
    arena_dealloc_to(tmp->arena, tmp->start_pos);
}
