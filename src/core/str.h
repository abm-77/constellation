#pragma once

#include "language_layer.h"
#include "mem.h"
#include <stdio.h>

// heap allocated string
typedef struct String {
    char* txt;
    u64 length;
} String;

typedef struct StringBuilderNode {
    String str;
    struct StringBuilderNode* next;
} StringBuilderNode;

typedef struct StringBuilder {
    StringBuilderNode* start;
    StringBuilderNode* end;

    Arena* arena;
    u32 n_nodes;
    u32 size;
} StringBuilder;

String string_create(Arena* arena, char* txt) {
    String str = {0};
    str.length = strlen(txt);
    str.txt = arena_push_array(arena, char, str.length);
    MemoryCopy(str.txt, txt, str.length);
    return str;
}

void string_print(String* str) {
    printf("%s\n", str->txt);
}

StringBuilder string_builder_create(Arena* arena) {
    StringBuilder sb =  {
        .start = NULL,
        .end = NULL,
        .arena = arena,
    };
    return sb;
}

void string_builder_append(StringBuilder* sb, char* str) {
    StringBuilderNode* node = arena_push(sb->arena, StringBuilderNode); 
    node->str = string_create(sb->arena, str);
    node->next = NULL;

    if (sb->n_nodes == 0) {
        sb->start = node;
        sb->end = node;
    }
    else {
        sb->end->next = node;
        sb->end = node;
    }

    sb->n_nodes += 1;
    sb->size += node->str.length;
}

String string_builder_build(StringBuilder* sb) {
    u32 offset = 0;
    char* buff = arena_push_array(sb->arena, char, sb->size);

    for (StringBuilderNode* node = sb->start; node != NULL; node = node->next) {
        MemoryCopy(buff + offset, node->str.txt, node->str.length);
        offset += node->str.length;
    }
    
    String str = {
        .txt = buff,
        .length = sb->size,
    };
    return str;
}














