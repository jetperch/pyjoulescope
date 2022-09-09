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

// gcc -Wall -pedantic -o test_filter_fir.exe ../filter_fir.c test_filter_fir.c && ./test_filter_fir.exe

#include "../filter_fir.h"
#include "acutest.h"
#include <math.h>
#include <stdio.h>

#define ARRAY_SIZE(x) (sizeof(x) / sizeof(x[0]))

const double taps_ma3[3] = {1.0/3, 1.0/3, 1.0/3};
double data_x1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
double data_y1[] = {0, 1.0/3, 1, 2, 3, 4, 5, 6, 7, 8, 9};
double data_y2[] = {1.0/3, 2, 4, 6, 8};


uint32_t data_offset;
struct filter_fir_s * f;


void cbk1(void * user_data, double const * x, uint32_t x_length) {
    TEST_ASSERT(user_data == (void *) &data_offset);
    TEST_ASSERT(x_length == 1);
    TEST_ASSERT(fabs(data_y1[data_offset++] - *x) < 1e-6);
}

void test_no_downsample(void) {
    data_offset = 0;
    f = filter_fir_alloc(taps_ma3, ARRAY_SIZE(taps_ma3), 1, 1);
    TEST_ASSERT(f != 0);
    filter_fir_callback_set(f, cbk1, &data_offset);
    for (int i = 0; i < ARRAY_SIZE(data_x1); ++i) {
        filter_fir_single(f, &data_x1[i], 1);
    }
    filter_fir_free(f);
}


void cbk2(void * user_data, double const * x, uint32_t x_length) {
    TEST_ASSERT(fabs(data_y2[data_offset++] - *x) < 1e-6);
}

void test_downsample_by_2(void) {
    data_offset = 0;
    f = filter_fir_alloc(taps_ma3, ARRAY_SIZE(taps_ma3), 2, 1);
    TEST_ASSERT(f != 0);
    filter_fir_callback_set(f, cbk2, &data_offset);
    for (int i = 0; i < ARRAY_SIZE(data_x1); ++i) {
        filter_fir_single(f, &data_x1[i], 1);
    }
    filter_fir_free(f);
}


void cbk3(void * user_data, double const * x, uint32_t x_length) {
    TEST_ASSERT(x_length == 2);
    TEST_ASSERT(fabs(data_y1[data_offset++] - x[0]) < 1e-6);
    TEST_ASSERT(fabs(data_y1[data_offset++] - x[1]) < 1e-6);
}

void test_downsample_by_2_width_2(void) {
    data_offset = 1;
    f = filter_fir_alloc(taps_ma3, ARRAY_SIZE(taps_ma3), 2, 2);
    TEST_ASSERT(f != 0);
    filter_fir_callback_set(f, cbk3, &data_offset);
    for (int i = 0; i < ARRAY_SIZE(data_x1) - 1; ++i) {
        filter_fir_single(f, &data_x1[i], 2);
    }
    filter_fir_free(f);
}


TEST_LIST = {
    { "no_downsample", test_no_downsample },
    { "downsample_by_2", test_downsample_by_2 },
    { "downsample_by_2_width_2", test_downsample_by_2_width_2 },
    { NULL, NULL }
};
