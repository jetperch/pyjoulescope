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


#include "filter_fir.h"
#include <stdlib.h>


struct filter_fir_s {
    double const * taps;
    uint32_t taps_length;
    double * buffer;
    double * buffer_end;
    double * y;  // accumulator y value computation
    uint32_t buffer_offset;
    uint32_t M;
    uint32_t M_counter;
    uint32_t width;

    filter_fir_cbk cbk_fn;
    void * cbk_user_data;
};

struct filter_fir_s * filter_fir_alloc(double const * taps, uint32_t taps_length,
        uint32_t M, uint32_t width) {
    struct filter_fir_s * self = malloc(sizeof(struct filter_fir_s));
    if (!self) {
        return 0;
    }
    self->buffer = 0;
    self->y = 0;
    self->cbk_fn = 0;
    self->cbk_user_data = 0;
    self->taps = taps;
    self->taps_length = taps_length;
    self->M = M;
    self->width = width;
    self->buffer = malloc(sizeof(double) * taps_length * width);
    self->y = malloc(sizeof(double) * width);
    if (!self->buffer || ! self->y) {
        filter_fir_free(self);
        return 0;
    }
    self->buffer_end = self->buffer + taps_length * width;
    filter_fir_reset(self);
    return self;
}

void filter_fir_free(struct filter_fir_s * self) {
    if (self) {
        if (self->buffer) {
            free(self->buffer);
        }
        if (self->y) {
            free(self->y);
        }
        free(self);
    }
}

void filter_fir_reset(struct filter_fir_s * self) {
    double * b = self->buffer;
    while (b < self->buffer_end) {
        *b++ = 0.0;
    }
    for (uint32_t i = 0; i < self->width; ++i) {
        self->y[i] = 0.0;
    }
    self->buffer_offset = 0;
    self->M_counter = 0;
}

void filter_fir_callback_set(struct filter_fir_s * self, filter_fir_cbk fn, void * user_data) {
    self->cbk_fn = fn;
    self->cbk_user_data = user_data;
}

void filter_fir_single(struct filter_fir_s * self, double const * x, uint32_t x_length) {
    double const * taps;
    double const * taps_end;
    double y;
    double * buffer;
    const uint32_t width = self->width;

    // add to buffer
    double * b = &self->buffer[self->buffer_offset * width];
    for (uint32_t idx = 0; idx < width; ++idx) {
        *b++ = *x++;
    }

    if (++self->M_counter >= self->M) {
        // downsampled interval, compute filtered output
        buffer = self->buffer;
        taps_end = self->taps + self->taps_length;
        for (uint32_t idx = 0; idx < width; ++idx) {
            y = 0.0;
            taps = self->taps;
            b = &buffer[self->buffer_offset * width + idx];
            while (b >= buffer) {
                y += *b * *taps++;
                b -= width;
            }
            b += (self->taps_length) * width;
            while (taps < taps_end) {
                y += *b * *taps++;
                b -= width;
            }
            self->y[idx] = y;
        }

        self->M_counter = 0;
        if (self->cbk_fn) {
            self->cbk_fn(self->cbk_user_data, self->y, width);
        }
    }

    if (++self->buffer_offset >= self->taps_length) {
        self->buffer_offset = 0;
    }
}
