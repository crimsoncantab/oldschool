#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define NTHREADS 8
#define NCPUS 8
#define PAGESIZE 4096
#define OPS (1 << 27)
#define COMPILE_TIME_ASSERT(pred) switch(0){case 0:case pred:;}

struct pad_ {
	char pad[PAGESIZE];
};

typedef struct raw_pg_ {
	long val;
} raw_pg_t;

typedef struct plock_pg_ {
	pthread_mutex_t lock;
} plock_pg_t;

typedef struct slock_pg_ {
	volatile unsigned int lock;
} slock_pg_t;

typedef union page_ {
	raw_pg_t p_raw;
	plock_pg_t p_plock;
	slock_pg_t p_slock;
	struct pad_ p_pad;
} page_t; //sizeof(page) == 4096

typedef struct work_ {
	page_t * w_mem;
	int w_tid;
	char w_pad[116]; //so they appear on different cache lines
} work_t; //sizeof(stride_t) = 128

typedef enum affin_ {
	NONE = 0,
	SAME = 1,
	DIFF = 2,
	ODD = 3,
	EVEN = 4,
	ONE = 5

} affin_t;

typedef struct benchmark_ {
	void * (*b_run)(work_t * work);
	void (*b_setup) ();
	void (*b_tear) ();
} benchmark_t;

void
alloc_mem();

void
free_mem();

void
plock_setup();

void
slock_setup();

void *
b_write(work_t * ptr);

void *
b_read(work_t * ptr);

void *
b_plock(work_t * ptr);

void *
b_slock(work_t * ptr);

static benchmark_t benchmarks[] = {
	{&b_write, &alloc_mem, &free_mem},
	{&b_read, &alloc_mem, &free_mem},
	{&b_plock, &plock_setup, &free_mem},
	{&b_slock, &slock_setup, &free_mem},
};

benchmark_t benchmark;
int local;
int write;
affin_t affin_mode;
long npages;
long nstrides;
int go = 0;

work_t workers[NTHREADS];

static long long
timeval_diff(struct timeval *difference,
		struct timeval *end_time,
		struct timeval *start_time
		) {
	struct timeval temp_diff;

	if (difference == NULL) {
		difference = &temp_diff;
	}

	difference->tv_sec = end_time->tv_sec - start_time->tv_sec;
	difference->tv_usec = end_time->tv_usec - start_time->tv_usec;

	/* Using while instead of if below makes the code slightly more robust. */

	while (difference->tv_usec < 0) {
		difference->tv_usec += 1000000;
		difference->tv_sec -= 1;
	}

	return 1000000LL * difference->tv_sec +
			difference->tv_usec;

}

static void
handle_args(int argc, char *argv[]) {
	benchmark = benchmarks[(argc > 1) ? atoi(argv[1]) : 0]; //QUITE UNSAFE
	npages = (argc > 2) ? 1 << atoi(argv[2]) : 256;
	affin_mode = (argc > 3) ? (affin_t) atoi(argv[3]) : ONE;
	local = (affin_mode != ONE && argc > 4) ? atoi(argv[4]) : 0;
	nstrides = (OPS) / npages;
}

void *
dispatch(void * ptr) {
	work_t * work = ptr;

	while (!go)
		/* do nothing */;

	return benchmark.b_run(work);
}

void
assign_thread(pthread_t thread, int tid) {
	cpu_set_t cpuset;
	int r, j;
	workers[tid].w_tid = tid;
	CPU_ZERO(&cpuset);
	switch (affin_mode) {
		case SAME:
			CPU_SET(1, &cpuset); //use CPU 1
			break;
		case DIFF:
			CPU_SET(tid, &cpuset);
			break;
		case ODD:
			CPU_SET(tid | 1, &cpuset);
			break;
		case EVEN:
			CPU_SET(tid&~1, &cpuset);
			break;
		case NONE:
		default:
			for (j = 0; j < NCPUS; j++) {
				CPU_SET(j, &cpuset);
			}
			break;
	}
	if (r = pthread_setaffinity_np(thread, sizeof (cpu_set_t), &cpuset)) {
		printf("ERR %d\n", r);
	}

}

int
main(int argc, char *argv[]) {
	COMPILE_TIME_ASSERT(sizeof (work_t) == 128);
	COMPILE_TIME_ASSERT(sizeof (page_t) == PAGESIZE);
	pthread_t threads[NTHREADS];
	int i;
	struct timeval begin, end, diff;
	
	handle_args(argc, argv);
	benchmark.b_setup();
	if (affin_mode == ONE) {
		gettimeofday(&begin, NULL);
		benchmark.b_run(&workers[0]);
		gettimeofday(&end, NULL);
	} else {
		for (i = 0; i < NTHREADS; i++) {
			pthread_create(&threads[i], NULL, dispatch, &workers[i]);
			assign_thread(threads[i], i);
		}
		gettimeofday(&begin, NULL);
		go = 1;
		for (i = 0; i < NTHREADS; i++) {
			pthread_join(threads[i], NULL);
		}
		gettimeofday(&end, NULL);
	}

	benchmark.b_tear();
	timeval_diff(&diff, &end, &begin);
	printf("%ld %ld.%ld\n", npages, diff.tv_sec, diff.tv_usec / 1000);

}

void
alloc_mem() {
	int i;
	page_t * array;
	if (local) {
		for (i = 0; i < NTHREADS; i++) {
			workers[i].w_mem = (page_t *) malloc(npages * sizeof (page_t));
		}
	} else {
		array = (page_t *) malloc(npages * sizeof (page_t));
		for (i = 0; i < NTHREADS; i++) {
			workers[i].w_mem = array;
		}
	}
}

void
free_mem() {
	int i;
	if (local) {
		for (i = 0; i < NTHREADS; i++) {
			free(workers[i].w_mem);
		}
	} else {
		free(workers[0].w_mem);
	}
}

void
plock_setup() {
	int i, j;
	alloc_mem();
	if (local) {
		for (i = 0; i < NTHREADS; i++) {
			for (j = 0; j < npages; j++) {
				pthread_mutex_init(&(workers[i].w_mem[j].p_plock.lock), NULL);
			}
		}
	} else {
		for (j = 0; j < npages; j++) {
			pthread_mutex_init(&(workers[0].w_mem[j].p_plock.lock), NULL);
		}
	}
}

void
slock_setup() {
	int i, j;
	alloc_mem();
	if (local) {
		for (i = 0; i < NTHREADS; i++) {
			for (j = 0; j < npages; j++) {
				workers[i].w_mem[j].p_slock.lock = 0;
			}
		}
	} else {
		for (j = 0; j < npages; j++) {
			workers[0].w_mem[j].p_slock.lock = 0;
		}
	}
}

void *
b_write(work_t * ptr) {
	long i, j, str = nstrides, n = npages;
	page_t * arr = ptr->w_mem;

	for (i = 0; i < str; i++) {
		for (j = 0; j < n; j++) {
			arr[j].p_raw.val = i;
		}
	}
	return NULL;
}

void *
b_read(work_t * ptr) {
	long i, j, ret, str = nstrides, n = npages;
	page_t * arr = ptr->w_mem;

	for (i = 0; i < str; i++) {
		for (j = 0; j < n; j++) {
			ret += arr[j].p_raw.val;
		}
	}
	return NULL;
}

void *
b_plock(work_t * ptr) {
	long i, j, ret = 0, str = nstrides;
	page_t * arr = (page_t *) ptr->w_mem;
	for (i = 0; i < str; i++) {
		for (j = 0; j < npages; j++) {
			pthread_mutex_lock(&arr[j].p_plock.lock);
			ret = i;
			pthread_mutex_unlock(&arr[j].p_plock.lock);
		}
	}
	return (void *) ret;
}

static inline unsigned int
xchg(volatile unsigned int *addr, unsigned int newval) {
	unsigned int result;

	// The + in "+m" denotes a read-modify-write operand.
	asm volatile("lock; xchgl %0, %1" :
			"+m" (*addr), "=a" (result) :
			"1" (newval) :
			"cc");
	return result;
}

static inline void
spin_lock(volatile unsigned int * lock) {
	while (xchg(lock, 1))
		asm volatile ("pause");
}

static inline void
spin_unlock(volatile unsigned int *lock) {
	xchg(lock, 0);
}

void *
b_slock(work_t * ptr) {
	long i, j, ret = 0, str = nstrides;
	page_t * arr = (page_t *) ptr->w_mem;
	for (i = 0; i < str; i++) {
		for (j = 0; j < npages; j++) {
			spin_lock(&(arr[j].p_slock.lock));
			ret = i;
			spin_unlock(&(arr[j].p_slock.lock));
		}
	}
	return (void *) ret;
}

