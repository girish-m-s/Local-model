/* Pure-serial STREAM triad — MUST be compiled WITHOUT -fopenmp.
 * Ground-truth baseline for DIAG A (A1). No OpenMP pragmas.
 * Inner loop body must match kernels.c triad for fair comparison.
 */
#include <stdint.h>
#include <stddef.h>
#include <sched.h>

void stream_triad_reps_pure_serial(double *a, const double *b, const double *c,
                                   double scalar, size_t n, size_t reps) {
    for (size_t r = 0; r < reps; r++) {
        for (size_t i = 0; i < n; i++) {
            a[i] = b[i] + scalar * c[i];
        }
    }
}

/* Instrumented serial: reports thread_count=1 (no OpenMP) + sched_getcpu sample. */
void stream_triad_reps_pure_serial_observed(
    double *a, const double *b, const double *c, double scalar,
    size_t n, size_t reps,
    int *out_omp_num_threads,
    int *out_omp_max_threads,
    int *out_n_distinct_cpus,
    int *out_cpu_ids,
    int cpu_ids_cap
) {
    int cpu = sched_getcpu();
    for (size_t r = 0; r < reps; r++) {
        for (size_t i = 0; i < n; i++) {
            a[i] = b[i] + scalar * c[i];
        }
    }
    if (out_omp_num_threads) *out_omp_num_threads = 1; /* serial — not from OpenMP */
    if (out_omp_max_threads) *out_omp_max_threads = 1;
    if (out_n_distinct_cpus) *out_n_distinct_cpus = 1;
    if (out_cpu_ids && cpu_ids_cap > 0) {
        out_cpu_ids[0] = cpu;
    }
}
