/* Phase 2 synthetic kernels — compiled at runtime with -O3 -fopenmp */
#include <stdint.h>
#include <stddef.h>

void stream_triad(double *a, const double *b, const double *c, double scalar, size_t n) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
        a[i] = b[i] + scalar * c[i];
    }
}

void stream_triad_serial(double *a, const double *b, const double *c, double scalar, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] = b[i] + scalar * c[i];
    }
}

/* Fixed-reps STREAM triad (constant work across thread sweep). */
void stream_triad_reps(double *a, const double *b, const double *c, double scalar,
                       size_t n, size_t reps) {
    for (size_t r = 0; r < reps; r++) {
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) {
            a[i] = b[i] + scalar * c[i];
        }
    }
}

void stream_triad_reps_serial(double *a, const double *b, const double *c, double scalar,
                              size_t n, size_t reps) {
    for (size_t r = 0; r < reps; r++) {
        for (size_t i = 0; i < n; i++) {
            a[i] = b[i] + scalar * c[i];
        }
    }
}

/* FP32 accumulate: many passes over L2-sized vectors (compute-ish). */
double dot_fp32_reps(const float *a, const float *b, size_t n, size_t reps) {
    double acc = 0.0;
    for (size_t r = 0; r < reps; r++) {
        float local = 0.0f;
#pragma omp parallel for reduction(+:local) schedule(static)
        for (size_t i = 0; i < n; i++) {
            local += a[i] * b[i];
        }
        acc += (double)local;
    }
    return acc;
}

double dot_fp32_reps_serial(const float *a, const float *b, size_t n, size_t reps) {
    double acc = 0.0;
    for (size_t r = 0; r < reps; r++) {
        float local = 0.0f;
        for (size_t i = 0; i < n; i++) {
            local += a[i] * b[i];
        }
        acc += (double)local;
    }
    return acc;
}

/* int8 → int32 accumulate (GOPS counting as 2 ops per multiply-add pair? we count 1 mul+1 add = 2) */
int64_t dot_i8_reps(const int8_t *a, const int8_t *b, size_t n, size_t reps) {
    int64_t acc = 0;
    for (size_t r = 0; r < reps; r++) {
        int32_t local = 0;
#pragma omp parallel for reduction(+:local) schedule(static)
        for (size_t i = 0; i < n; i++) {
            local += (int32_t)a[i] * (int32_t)b[i];
        }
        acc += local;
    }
    return acc;
}

int64_t dot_i8_reps_serial(const int8_t *a, const int8_t *b, size_t n, size_t reps) {
    int64_t acc = 0;
    for (size_t r = 0; r < reps; r++) {
        int32_t local = 0;
        for (size_t i = 0; i < n; i++) {
            local += (int32_t)a[i] * (int32_t)b[i];
        }
        acc += local;
    }
    return acc;
}

/* Touch one byte per page (or stride) to force major/minor faults. */
size_t stride_touch(const char *p, size_t nbytes, size_t stride) {
    size_t acc = 0;
    if (stride == 0) stride = 4096;
    for (size_t i = 0; i < nbytes; i += stride) {
        acc += (unsigned char)p[i];
    }
    return acc;
}

/* Same, but takes an integer address (for read-only mmap via numpy.ctypes.data). */
size_t stride_touch_addr(uintptr_t addr, size_t nbytes, size_t stride) {
    return stride_touch((const char *)addr, nbytes, stride);
}
