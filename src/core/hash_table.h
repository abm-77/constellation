#pragma once

#include "language_layer.h"
#include "str.h"

#define DEFAULT_NUM_SLOTS = 256;

// djb2 hash
u64 hash_string(char* str) {
    u64 result = 5381;
    for (i32 i = 0; i < strlen(str); i++) result = ((result << 5) + result) + str[i];
    return result;
}

typedef struct HashNode {
    char* key;
    byte* value;
    u64 hash;

    struct HashNode* next;
} HashNode;

typedef struct HashTable {
    Arena* arena;
    HashNode** slots;
    u64 slot_count;
} HashTable;

HashTable hash_table_create(Arena* arena, u64 slot_count) {
    HashTable ht = {
        .arena = arena,
        .slots = arena_push_array(arena, HashNode*, slot_count),
        .slot_count = slot_count,
    };
    return ht;
}

byte* hash_table_get(HashTable* ht, char* key) {
    byte* result = NULL;

    u64 hash = hash_string(key);

    u64 slot_idx = hash % ht->slot_count;

    HashNode* node = NULL;
    for (HashNode* curr = ht->slots[slot_idx]; curr != NULL; curr = curr->next) {
        if(curr->hash == hash) {
            node = curr;
            break;
        }
    }

    if (node != NULL) result = node->value;

    return result;
}

void hash_table_insert(HashTable* ht, char* key, byte* value) {
    u64 hash = hash_string(key);
    u64 slot_idx = hash % ht->slot_count;

    HashNode* existing_node = NULL;
    for (HashNode* curr = ht->slots[slot_idx]; curr != NULL; curr = curr->next) {
        if(curr->hash == hash) {
            existing_node = curr;
            break;
        }
    }

    if (existing_node == NULL) {
        HashNode* new_node = arena_push(ht->arena, HashNode);
        new_node->key = key;

        new_node->value = value;

        new_node->hash = hash;
        new_node->next = ht->slots[slot_idx];
        ht->slots[slot_idx] = new_node;
    }
    else {
        existing_node->value = value;
    }
}

typedef struct HashSet {
    HashTable ht;
} HashSet;

HashSet hash_set_create(Arena* arena, u64 slot_count) {
    HashSet hs = {
        .ht = hash_table_create(arena, slot_count)
    };
    return hs;
}

void hash_set_insert(HashSet* hs, char* key) {
    hash_table_insert(&hs->ht, key, (byte*) 1);
}

void hash_set_remove(HashSet* hs, char* key) {
    hash_table_insert(&hs->ht, key, (byte*) 0);
}

b32 hash_set_has(HashSet* hs, char* key) {
    byte* result = hash_table_get(&hs->ht, key);
    return result != NULL && result == (byte*) 1;
}
