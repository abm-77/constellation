#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <sqlite3.h>

#include "str.h"

#define SQL_CHECK(db,rc,msg)\
    do {\
        if (rc != SQLITE_OK) {\
            printf("%s\n", msg);\
            printf("SQL Error: %s\n", sqlite3_errmsg(db));\
            sqlite3_close(db);\
            abort();\
        }\
    } while(0)\

#define SQL_CHECK_ERR(db,rc,err_msg)\
    do {\
        if (rc != SQLITE_OK) {\
            printf("SQL Error: %s\n", err_msg);\
            sqlite3_free(err_msg);\
            sqlite3_close(db);\
            abort();\
        }\
    } while(0)\

#define sql_open(db,name)\
    SQL_CHECK((db)->db_ptr, sqlite3_open(name, &((db)->db_ptr)), "Cannot open database!")\

#define sql_exec(db,cmd)\
    SQL_CHECK_ERR((db)->db_ptr, sqlite3_exec((db)->db_ptr, cmd->sql_code.txt, 0, 0, &((db)->err_msg)), db->err_msg)\

#define sql_exec_callback(db,cmd,cb)\
    SQL_CHECK_ERR((db)->db_ptr, sqlite3_exec((db)->db_ptr, cmd->sql_code.txt, cb, 0, &((db)->err_msg)), db->err_msg)\

#define sql_prepare(db,cmd,msg)\
    SQL_CHECK((db)->db_ptr, sqlite3_prepare_v2((db)->db_ptr, cmd->sql_code.txt, -1, &((db)->res), 0), msg)\

typedef struct SQLDB {
    sqlite3* db_ptr;
    char* err_msg;
    sqlite3_stmt* res;
} SQLDB;

typedef struct SQLCommandBuffer {
    StringBuilder sb;
} SQLCommandBuffer;

typedef struct SQLCommand {
    String sql_code;
} SQLCommand;

SQLDB sql_db_create(char* db_name) {
    SQLDB db = {0};
    sql_open(&db, db_name);
    return db;
}

void sql_db_close(SQLDB* db) {
    sqlite3_close(db->db_ptr);
}

void sql_db_submit(SQLDB* db, SQLCommand* cmd) {
    sql_exec(db, cmd);
}

void sql_db_prepare(SQLDB* db, SQLCommand* cmd) {
    sql_prepare(db,cmd,"error");
}

i32 sql_db_step(SQLDB* db) {
    return sqlite3_step(db->res); 
}

SQLCommandBuffer sql_command_buffer_begin(Arena* arena) {
    SQLCommandBuffer cmd_buff = {
        .sb = string_builder_create(arena),
    };
    return cmd_buff;
}

void sql_command_buffer_push(SQLCommandBuffer* cmd_buff, char* cmd) {
    string_builder_append(&cmd_buff->sb, cmd);
}

SQLCommand sql_command_create(Arena* arena, char* cmd_body) {
    SQLCommand cmd = {
        .sql_code = string_create(arena, cmd_body),
    };
    return cmd;
}

SQLCommand sql_command_buffer_end(SQLCommandBuffer* cmd_buff) {
    SQLCommand cmd = {
         .sql_code = string_builder_build(&cmd_buff->sb),
    };
    return cmd;
}






