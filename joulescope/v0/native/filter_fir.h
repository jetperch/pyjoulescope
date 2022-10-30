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

#include <stdint.h>


/// Opaque filter instance handle.
struct filter_fir_s;


/**
 * @brief The callback invoked for each processed filter output sample.
 *
 * @param user_data The arbitrary user data provided to filter_fir_callback_set().
 * @param y The output data.
 * @param y_length The length of y which matches the width provided to
 *      filter_fir_alloc().
 *
 * The number of callbacks will be filter_fir_single() / M.
 */
typedef void (*filter_fir_cbk)(void * user_data, double const * y, uint32_t y_length);

/**
 * @brief Create a new FIR filter instance intended for data streaming.
 *
 * @param taps The FIR filter taps.  The taps pointer must remain valid until
 *      filter_fir_free().
 * @param taps_length The number of double values in taps (also filter order - 1).
 * @param M The decimation factor.  M=1 does not decimate.  M=2 discards every
 *      other sample.
 * @param width The number of independent data samples to process together.
 *      width=1 is a normal 1D filter.  width=2 processes two signals
 *      simultaneously.
 * @return A new FIR filter instance.  When done, call filter_fir_free().
 */
struct filter_fir_s * filter_fir_alloc(double const * taps, uint32_t taps_length,
    uint32_t M, uint32_t width);

/**
 * @brief Free an FIR filter instance.
 *
 * @param self The FIR filter instance to free.
 *      The instance must not be used after free.
 */
void filter_fir_free(struct filter_fir_s * self);

/**
 * @brief Reset the filter instance and clear all data.
 *
 * @param The FIR filter instance.
 */
void filter_fir_reset(struct filter_fir_s * self);

/**
 * @brief Set the callback for filtered data.
 *
 * @param The FIR filter instance.
 * @param fn The function called with each sample.
 * @param user_data The arbitrary data for cbk_fn.
 */
void filter_fir_callback_set(struct filter_fir_s * self, filter_fir_cbk fn, void * user_data);

/**
 * @brief Process a single sample.
 *
 * @param self The FIR filter instance.
 * @param x The data sample for each signal.
 * @param x_length The length of x, which must match the width provided to
 *      filter_fir_alloc().
 */
void filter_fir_single(struct filter_fir_s * self, double const * x, uint32_t x_length);
