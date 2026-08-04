/* Pure-serial STREAM triad — MUST be compiled WITHOUT -fopenmp.
 * Ground-truth baseline for Phase 2.2 DIAG A (A1). No OpenMP pragmas.
 */
#include <stdint.h>
#include <stddef.h>

void stream_triad_reps_pure_serial(double *a, const double *b, const double *c,
                                   double scalar, size_t n, size_t reps) {
    for (size_t r = 0; r < reps; r++) {
        for (size_t i = 0; i < n; i++) {
            a[i] = b[i] + scalar * c[i];
        }
    }
}
