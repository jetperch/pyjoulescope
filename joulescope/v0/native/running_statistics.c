/*
 * Copyright 2020 Jetperch LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "running_statistics.h"
#include <float.h>
#include <math.h>

void statistics_reset(struct statistics_s * s) {
    s->k = 0;
    s->m = 0.0;
    s->s = 0.0;
    s->min = DBL_MAX;
    s->max = -DBL_MAX;
};

void statistics_invalid(struct statistics_s * s) {
    s->m = NAN;
    s->s = NAN;
    s->min = NAN;
    s->max = NAN;
}

void statistics_add(struct statistics_s * s, double x) {
    double m_old;
    double m_new;
    ++s->k;
    m_old = s->m;
    m_new = s->m + (x - s->m) / (double) s->k;
    s->m = m_new;
    s->s += (x - m_old) * (x - m_new);
    if (x < s->min) {
        s->min = x;
    }
    if (x > s->max) {
        s->max = x;
    }
}

double statistics_var(struct statistics_s * s) {
    if (s->k <= 1) {
        return 0.0;
    }
    return s->s / (double) (s->k - 1); // use k - 1 = Bessel's correction
}

void statistics_copy(struct statistics_s * tgt, 
                     struct statistics_s const * src) {
    tgt->k = src->k;
    tgt->m = src->m;
    tgt->s = src->s;
    tgt->min = src->min;
    tgt->max = src->max;
};

void statistics_combine(struct statistics_s * tgt, 
                        struct statistics_s const * a, 
                        struct statistics_s const * b) {
    uint64_t kt;
    double f1;
    double f2;
    double m1_diff;
    double m2_diff;
    kt = a->k + b->k;
    if (kt == 0) {
        ; // pass
    } else if (a->k == 0) {
        statistics_copy(tgt, b);
    } else if (b->k == 0) {
        statistics_copy(tgt, a);
    } else {
        f1 = a->k / (double) kt;
        f2 = 1.0 - f1;
        double mean_new = f1 * a->m + f2 * b->m;
        m1_diff = a->m - mean_new;
        m2_diff = b->m - mean_new;
        tgt->s = ((a->s + a->k * m1_diff * m1_diff) +
                  (b->s + b->k * m2_diff * m2_diff));
        tgt->m = mean_new;
        tgt->min = (a->min < b->min) ? a->min : b->min;
        tgt->max = (a->max > b->max) ? a->max : b->max;
        tgt->k = kt;
    }
}
