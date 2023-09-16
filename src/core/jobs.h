#include <pthread.h>
#include <stdlib.h>
#include <sched.h>
#include <unistd.h>
#include <stdio.h>
#include <stdatomic.h>

#include <core/language_layer.h>
#include <core/rand.h>
#include <core/mem.h>

#define atomic_increment(pval) __atomic_fetch_add(pval, 1, __ATOMIC_SEQ_CST)
#define atomic_decrement(pval) __atomic_fetch_sub(pval, 1, __ATOMIC_SEQ_CST)
#define atomic_compare_exchange(pval,pexpected,desired)__atomic_compare_exchange_n(pval, pexpected, desired, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)
#define yield() sched_yield()

#define MAX_JOB_COUNT 256
#define MOD_MASK (MAX_JOB_COUNT - 1)

int get_available_cores() {
    int n_cpu = 0;
#if defined(_WIN32) || defined(_WIN64)
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    n_cpu = sysinfo.dwNumberOfProcessors;
#else
    n_cpu = sysconf(_SC_NPROCESSORS_ONLN);
#endif
    return n_cpu;
}

#define CACHE_SIZE 64
size_t get_cache_line_size() {
    size_t line_size = 0;
#if defined(_WIN32) || defined(_WIN64)
#else
#include<sys/sysctl.h>
    line_size = 0;
    size_t len = sizeof(line_size);
    sysctlbyname("hw.cachelinesize", &line_size, &len, 0, 0);
#endif
    return line_size;
}

#pragma region jobs
typedef struct Job Job;
typedef void (*JobFunc)(Job*, void*);

#define JOB_DATA_SIZE CACHE_SIZE  - (sizeof(JobFunc) + sizeof(Job*) + sizeof(volatile _Atomic(i32)))

typedef struct Job {
    JobFunc function;
    Job* parent;
    i32 unfinished_jobs;
    char data[JOB_DATA_SIZE];
} Job;

#pragma endregion

#pragma region job_pool
thread_local Job g_job_allocator[MAX_JOB_COUNT];
thread_local u32 g_allocated_jobs = 0;

Job* job_alloc() {
    // TODO(bryson): add checks to ensure more than MAX_JOB_COUNT jobs are not allocated per frame
    u32 idx = g_allocated_jobs++;
    return &g_job_allocator[idx & MOD_MASK];
}

Job* job_create(JobFunc function) {
    Job* job = job_alloc();
    MemoryZero(job, sizeof(Job));
    job->function = function;
    job->parent = NULL;
    job->unfinished_jobs = 1;
    return job;
}

Job* job_create_child(Job* parent, JobFunc function) {
    atomic_increment(&parent->unfinished_jobs);

    Job* job = job_alloc();
    MemoryZero(job, sizeof(Job));
    job->function = function;
    job->parent = parent;
    job->unfinished_jobs = 1;
    return job;
}

void job_write_data(Job* job, char* data, u32 size) {
    Assert(size < JOB_DATA_SIZE);
    MemoryCopy(job->data, data, size);
}

b32 job_empty(Job* job) {
    return job == NULL;
}

void job_finish(Job* job) {
    atomic_decrement(&job->unfinished_jobs);
    if ((job->unfinished_jobs == 0) && (job->parent)) {
        job_finish(job->parent);
    }
}

void job_execute(Job* job) {
    (job->function)(job, job->data);
    job_finish(job);
}

b32 job_has_completed(Job* job) {
    return job->unfinished_jobs == 0;
}
#pragma endregion

#pragma region job_queue
typedef struct JobQueue {
    Job* jobs[MAX_JOB_COUNT];
    u32 bottom;
    u32 top;
    pthread_mutex_t lock;
} JobQueue;

JobQueue job_queue_create() {
    JobQueue  queue;
    queue.bottom = 0;
    queue.top = 0;

    MemoryZero(queue.jobs, sizeof(Job*) * MAX_JOB_COUNT);

    i32 error = pthread_mutex_init(&queue.lock, NULL);
    if (error != 0) {
        printf("cannot create mutex: %s", strerror(error));
    }

    return queue;
}

b32 job_queue_push(JobQueue* queue, Job* job) {
    pthread_mutex_lock(&queue->lock);
    b32 pushed = true;
    if (queue->bottom - queue->top < MAX_JOB_COUNT) {
        queue->jobs[queue->bottom & MOD_MASK] = job;
        ++queue->bottom;
    }
    else {
        pushed = false;
    }
    pthread_mutex_unlock(&queue->lock);

    return pushed;
}

Job* job_queue_pop(JobQueue* queue) {
    pthread_mutex_lock(&queue->lock);
    Job* job = NULL;

    i32 job_count = queue->bottom - queue->top;
    if (job_count > 0)  {
        --queue->bottom;
        job = queue->jobs[queue->bottom & MOD_MASK];
    }

    pthread_mutex_unlock(&queue->lock);
    return job;
}

Job* job_queue_steal(JobQueue* queue) {
    pthread_mutex_lock(&queue->lock);
    Job* job = NULL;
    i32 job_count = queue->bottom - queue->top;
    if (job_count > 0) {
        job = queue->jobs[queue->top & MOD_MASK];
        ++queue->top;
    }
    pthread_mutex_unlock(&queue->lock);
    return job;
}
#pragma endregion

#pragma region workers
typedef struct Worker {
    pthread_t thread_id;
    JobQueue queue;
} Worker;

static struct {
    Worker* workers;
    u32 n_workers;
    Arena arena;

    pthread_mutex_t suspend_mutex;
    pthread_cond_t  resume_cond;
} _job_system;

void* worker_proc(void* arg);
void job_system_init() {
    _job_system.arena = arena_create(Megabytes(4));
    _job_system.n_workers = get_available_cores();
    _job_system.workers = arena_push_array(&_job_system.arena, Worker, _job_system.n_workers);

    pthread_mutex_init(&_job_system.suspend_mutex, NULL);
    pthread_cond_init(&_job_system.resume_cond, NULL);

    for (int i = 0; i < _job_system.n_workers; ++i) {
        Worker* worker = &_job_system.workers[i];
        worker->queue = job_queue_create();
        pthread_create(&worker->thread_id, NULL, worker_proc, (void*)worker);
        pthread_detach(worker->thread_id);
    }
}

Worker* job_system_get_random_worker() {
    u32 rand_idx = irand_range(0, _job_system.n_workers);
    return &_job_system.workers[rand_idx];
}

Worker* job_system_find_worker(pthread_t thread_id) {
    for (i32 i = 0; i < _job_system.n_workers; ++i) {
        if (pthread_equal(_job_system.workers[i].thread_id, thread_id)) {
            return &_job_system.workers[i];
        }
    }
    return NULL;
}

Worker* job_system_thread_worker() {
    for (i32 i = 0; i < _job_system.n_workers; ++i) {
        if (_job_system.workers[i].queue.bottom - _job_system.workers[i].queue.top < MAX_JOB_COUNT) {
            return &_job_system.workers[i];
        }
    }
    return NULL;
}

Job* worker_get_job(Worker* worker) {
    Job* job = job_queue_pop(&worker->queue);

    if (job_empty(job)) {
        // select random queue to steal from bc we got invalid job
        Worker* steal_worker = job_system_get_random_worker();

        // we don't want to steal from ourself, try again next time
        if (steal_worker == worker) {
            yield();
            return NULL;
        }
        
        Job* stolen_job = job_queue_steal(&steal_worker->queue);
        // stolen job was empty, try again next time
        if (job_empty(stolen_job)) {
            yield();
            return NULL;
        }
        return stolen_job;
    }

    return job;
}

void worker_wait(Worker* worker, Job* job) {
    while(!job_has_completed(job)) {
        Job* next_job = worker_get_job(worker);
        if (!job_empty(next_job)) {
            job_execute(next_job);
        }
    }
}

void worker_poll() {
    pthread_cond_signal(&_job_system.resume_cond);
    yield();
}

void worker_submit(Worker* worker, Job* job) {
    while(!job_queue_push(&worker->queue, job)) {
        worker_poll();
    };
    pthread_cond_signal(&_job_system.resume_cond);
}

void* worker_proc(void* arg) {
    for(;;) {
        Job* job = worker_get_job((Worker*) arg);
        if (!job_empty(job)) {
            job_execute(job);
        }
        else {
            pthread_mutex_lock(&_job_system.suspend_mutex);
            pthread_cond_wait(&_job_system.resume_cond, &_job_system.suspend_mutex);
            pthread_mutex_unlock(&_job_system.suspend_mutex);
        }
    }
}
#pragma endregion

#pragma region dispatch

typedef void (*ParFunc)(void*, u32);

#define PAR_GROUP_SIZE 32
typedef struct ParallelForData {
    void* data;
    u32 job_count;
    ParFunc par_func;
} ParallelForData;

void parallel_for_job(Job* job, void* data) {
    const ParallelForData* job_data = (ParallelForData*) data;

    if (job_data->job_count > PAR_GROUP_SIZE) {
        Worker* worker = job_system_thread_worker();

        const u32 left_count = job_data->job_count / 2u;
        const ParallelForData left_data = {
            .data = job_data->data,
            .job_count = left_count,
            .par_func = job_data->par_func,
        };

        Job* left = job_create_child(job, &parallel_for_job);
        job_write_data(left, (char*)&left_data, sizeof(ParallelForData));
        worker_submit(worker, left);

        const u32 right_count = job_data->job_count - left_count;
        const ParallelForData right_data = {
                .data = job_data->data + left_count,
                .job_count = right_count,
                .par_func = job_data->par_func,
        };

        Job* right = job_create_child(job, &parallel_for_job);
        job_write_data(right, (char*)&right_data, sizeof(ParallelForData));
        worker_submit(worker, right);
    }
    else {
        (job_data->par_func)(job_data->data, job_data->job_count);
    }
}

Job* parallel_for(void* data, u32 count, ParFunc par_func) {
    ParallelForData job_data = {
        .data = data,
        .job_count = count,
        .par_func = par_func,
    };

    Job* job = job_create(&parallel_for_job);
    job_write_data(job, (char*)&job_data, sizeof(ParallelForData));
    return job;
}

#pragma endregion

typedef struct Task {
    Worker* worker;
    Job* job;
} Task;

Task task_launch(JobFunc function) {
    Job* job = job_create(function);
    Worker* worker = job_system_thread_worker();
    worker_submit(worker, job);
    Task task = {.worker =worker, .job = job};
    return task;
}

void task_wait(Task* task) {
    worker_wait(task->worker, task->job);
}









