#ifndef BASE_TYPES_H
#define BASE_TYPES_H

#include <memory.h>
#include <stdint.h>
#include <stdlib.h>

/// NOTE(bryson): Context Cracking
#pragma region CONTEXT
#if defined(__clang__)
#define COMPILER_CLANG 1
#if defined(_WIN32)
#define OS_WINDOWS 1
#elif defined(__gnu_linux__)
#define OS_LINUX 1
#elif defined(__APPLE__) && defined(__MACH__)
#define OS_MAC 1
#else
#error Missing OS detection
#endif

#if defined(__amd64__)
#define ARCH_X64 1
#elif defined(__i386__)
#define ARCH_X86 1
#elif defined(__arm__)
#define ARCH_ARM 1
#elif defined(__arch64__)
#define ARCH_ARM64 1
#else
#error Missing ARCH detection
#endif

#elif defined(_MSC_VER)
#define COMPILER_CL 1
#if defined(_WIN32)
#define OS_WINDOWS 1
#else
#error Missing OS detection
#endif

#if defined(_M_AMD64)
#define ARCH_X64 1
#elif defined(_M_IX86)
#define ARCH_X86 1
#elif defined(_M_ARM)
#define ARCH_ARM 1
#else
#error Missing ARCH detection
#endif

#elif defined(__GNUC__)
#define COMPILER_GCC 1
#if defined(_WIN32)
#define OS_WINDOWS 1
#elif defined(__gnu_linux__)
#define OS_LINUX 1
#elif defined(__APPLE__) && defined(__MACH__)
#define OS_MAC 1
#else
#error Missing OS detection
#endif

#if defined(__amd64__)
#define ARCH_X64 1
#elif defined(__i386__)
#define ARCH_X86 1
#elif defined(__arm__)
#define ARCH_ARM 1
#elif defined(__arch64__)
#define ARCH_ARM64 1
#else
#error Missing ARCH detection
#endif
#else
#error Could not determine compiler!
#endif
#pragma endregion

// NOTE(bryson): Helper Macros
#pragma region HELPERS
#define Stmnt(S) \
  do { S } while (0)

#if !defined(AssertBreak)
#if defined(COMPILER_CLANG) || defined(COMPILER_GCC)
#define AssertBreak() __builtin_trap()
#else
#define AssertBreak() (*(int*)0 = 0)
#endif
#endif

#if ENABLE_ASSERT
#define Assert(c) Stmnt(if (!(c)) { AssertBreak(); })
#else
#define Assert(c)
#endif

#define _Stringify(S) #S
#define Stringify(S)  _Stringify(S)
#define _Glue(A, B)   A##B
#define Glue(A, B)    _Glue(A, B)

#define IntFromPtr(p) (unsigned long long)((char*)p - (char*)0)
#define PtrFromInt(i) (void*)((char*)0 + i)

#define Member(T, m)         (((T*)0)->m)
#define OffsetOfMember(T, m) IntFromPtr(&Member(T, m))

#define Min(a, b)      (((a) < (b)) ? (a) : (b))
#define Max(a, b)      (((a) > (b)) ? (a) : (b))
#define Sign(v)         ((0 < v) - (v < 0))
#define Clamp(l, x, u) (((x) < (l)) ? (l) : ((u) < (x)) ? (u) : (x))
#define ClampTop(a, b) Min(a, b)
#define ClampBot(a, b) Max(a, b)

#define AlignUpPow2(x, p) ((x) + (p)-1) & ~((p)-1)
#define AlignDown(x, p) ((x) & ~((p) - 1)

#define global static
#define local  static
#define func

#define CLinkageBegin extern "C" {
#define CLinkageEnd   }
#define CLinkage      extern "C"

#define MemoryMove       memmove
#define MemorySet        memset
#define MemoryCompare    memcmp
#define MemoryReallocate realloc

#define MemoryZero(p, z)      MemorySet((p), 0, (z))
#define MemoryZeroStruct(p)   MemoryZero(p, sizeof(*(p)))
#define MemoryZeroArray(p)    MemoryZero((p), sizeof(p))
#define MemoryZeroTyped(p, c) MemoryZero((p), sizeof(*(p)) * (c))

#define MemoryMatch(a, b, z) (MemoryCompare((a), (b), (z)) == 0)

#define MemoryCopy(d, s, z)      MemoryMove((d), (s), (z))
#define MemoryCopyStruct(d, s)   MemoryCopy((d), (s), Min(sizeof(*(d)), sizeof(*(s))))
#define MemoryCopyArray(d, s)    MemoryCopy((d), (s), Min(sizeof(s), sizeof(d)))
#define MemoryCopyTyped(d, s, c) MemoryCopy((d), (s), Min(sizeof(*(d)), sizeof(*(s)) * (c)))
#pragma endregion

// NOTE(bryson): Symbolic Constants
#pragma region SYMBOLIC_CONSTANTS
#define Bytes(n)     n
#define Kilobytes(n) (n << 10)
#define Megabytes(n) (n << 20)
#define Gigabytes(n) (((u64)n) << 30)
#define Terabytes(n) (((u64)n) << 40)

typedef enum Side { SIDE_MIN, SIDE_MAX } Side;

typedef enum OperatingSystem {
  OPERATINGSYSTEM_NULL,
  OPERATINGSYSTEM_WINDOWS,
  OPERATINGSYSTEM_MAC,
  OPERATINGSYSTEM_LINUX,
  OPERATINGSYSTEM_COUNT,
} OperatingSystem;

typedef enum Architecture {
  ARCHITECTURE_NULL,
  ARCHITECTURE_X86,
  ARCHITECTURE_X64,
  ARCHITECTURE_ARM,
  ARCHITECTURE_ARM64,
  ARCHITECTURE_COUNT,
} Architecture;

typedef enum Month {
  MONTH_JAN,
  MONTH_FEB,
  MONTH_MAR,
  MONTH_APR,
  MONTH_MAY,
  MONTH_JUN,
  MONTH_JUL,
  MONTH_AUG,
  MONTH_SEP,
  MONTH_OCT,
  MONTH_NOV,
  MONTH_DEC,
} Month;

typedef enum Weekday {
  WEEKDAY_MON,
  WEEKDAY_TUE,
  WEEKDAY_WED,
  WEEKDAY_THU,
  WEEKDAY_FRI,
  WEEKDAY_SAT,
  WEEKDAY_SUN,
} Weekday;

func OperatingSystem OperatingSystemFromContext(void);
func Architecture    ArchitectureFromContext(void);

func const char* StringFromOperatingSystem(OperatingSystem os);
func const char* StringFromArchitecture(Architecture arch);
func const char* StringFromMonth(Month month);
func const char* StringFromWeekday(Weekday day);
#pragma endregion

// NOTE(bryson): Types
#pragma region TYPES

#define true  1
#define false 0

typedef int8_t      i8;
typedef int16_t     i16;
typedef int32_t     i32;
typedef int64_t     i64;
typedef uint8_t     u8;
typedef uint16_t    u16;
typedef uint32_t    u32;
typedef uint64_t    u64;
typedef i8          s8;
typedef i16         s16;
typedef i32         s32;
typedef i64         s64;
typedef i8          b8;
typedef i16         b16;
typedef i32         b32;
typedef i64         b64;
typedef float       f32;
typedef double      f64;
typedef char        byte;
typedef const char* str;

typedef void VoidFunc(void);

// Basic Constants
global i8  MIN_I8  = (i8)0x80;
global i16 MIN_I16 = (i16)0x8000;
global i32 MIN_I32 = (i32)0x80000000;
global i64 MIN_I64 = (i64)0x8000000000000000llu;

global i8  MAX_I8  = (i8)0x7f;
global i16 MAX_I16 = (i16)0x7fff;
global i32 MAX_I32 = (i32)0x7fffffff;
global i64 MAX_I64 = (i64)0x7fffffffffffffffllu;

global i8  MAX_U8  = (i8)0xff;
global i16 MAX_U16 = (i16)0xffff;
global i32 MAX_U32 = (i32)0xffffffff;
global i64 MAX_U64 = (i64)0xffffffffffffffffllu;

global f32 MACHINE_EPSILON_F32 = 1.1920929e-07f;
global f32 PI_F32              = 3.14159265359f;
global f32 TAU_F32             = 6.28318530718f;
global f32 e_F32               = 2.71828182846f;
global f32 GOLD_BIG_F32        = 1.61803398875f;
global f32 GOLD_SMALL_F32      = 0.61803398875f;

global f64 MACHINE_EPSILON_F64 = 1.1920929e-07;
global f64 PI_F64              = 3.14159265359;
global f64 TAU_F64             = 6.28318530718;
global f32 e_F64               = 2.71828182846;
global f64 GOLD_BIG_F64        = 1.61803398875;
global f64 GOLD_SMALL_F64      = 0.61803398875;

#define ApproxLess(x, y)    ((x) < (y + MACHINE_EPSILON_F32))
#define ApproxGreater(x, y) ((x) > (y - MACHINE_EPSILON_F32))
#define ApproxEqual(x, y)   ApproxLess(x, y) && ApproxGreater(x, y)

func f32 LerpF32(f32 a, f32 b, f32 t);
func f32 UnlerpF32(f32 x, f32 a, f32 b);

func f32 INF_F32(void);
func f32 NEG_INF_F32(void);
func f64 INF_F64(void);
func f64 NEG_INF_F64(void);
#pragma endregion

// Note(bryson): Zero fill missing context macros
#pragma region DEFAULTS
#if !defined(COMPILER_CL)
#define COMPILER_CL 0
#endif
#if !defined(COMPILER_CLANG)
#define COMPILER_CLANG 0
#endif
#if !defined(COMPILER_GCC)
#define COMPILER_GCC 0
#endif
#if !defined(OS_WINDOWS)
#define OS_WINDOWS 0
#endif
#if !defined(OS_LINUX)
#define OS_LINUX 0
#endif
#if !defined(OS_MAC)
#define OS_MAC 0
#endif
#if !defined(ARCH_X64)
#define ARCH_X64 0
#endif
#if !defined(ARCH_X86)
#define ARCH_X86 0
#endif
#if !defined(ARCH_ARM)
#define ARCH_ARM 0
#endif
#if !defined(ARCH_ARM64)
#define ARCH_ARM64 0
#endif
#if !defined(ARCH_ARM64)
#define ARCH_ARM64 0
#endif
#if !defined(ENABLE_ASSERT)
#define ENABLE_ASSERT 0
#endif

#if !defined(DEBUG)
#define DEBUG 0
#endif

#pragma endregion

#define thread_local _Thread_local

#endif  // BASE_H
