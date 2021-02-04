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

#ifndef RUNNING_STATISTICS_H
#define RUNNING_STATISTICS_H

#include <stdint.h>


/**
 * @brief The statistics instance for a single variable.
 *
 * This structure and associated "methods" compute mean, sample variance, 
 * minimum and maximum over samples.  The statistics are computed in a single 
 * pass and are available at any time with minimal additional computation.
 *
 * @see https://en.wikipedia.org/wiki/Variance
 * @see https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation
 * @see https://www.johndcook.com/blog/standard_deviation/
 * @see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
 */
struct statistics_s {
    uint64_t k; // number of samples
    double m;   // mean
    double s;   // scaled running variance
    double min; // minimum value
    double max; // maximum value
};

/**
 * @brief Reset the statistics to 0 samples.
 *
 * @param s The statistics instance.
 */
void statistics_reset(struct statistics_s * s);

/**
 * @brief Mark the statistics as invalid.
 *
 * @param s The statistics instance which will have all statistics marked
 *      as NaN.
 */
void statistics_invalid(struct statistics_s * s);

/**
 * @brief Add a new sample into the statistics.
 *
 * @param s The statistics instance.
 * @param x The new value.
 */
void statistics_add(struct statistics_s * s, double x);

/**
 * @brief Get the sample variance.
 *
 * @param s The statistics instance.
 * @return The sample variance.
 */
double statistics_var(struct statistics_s * s);

/**
 * @brief Copy one statistics instance to another.
 *
 * @param tgt The target statistics instance.
 * @param src The source statistics instance.
 */
void statistics_copy(struct statistics_s * tgt, struct statistics_s const * src);

/**
 * @brief Compute the combined statistics over two statistics instances.
 *
 * @param tgt The target statistics instance.  It is safe to use a or b for tgt.
 * @param a The first statistics instance to combine.
 * @param b The first statistics instance to combine.
 */
void statistics_combine(struct statistics_s * tgt, 
                        struct statistics_s const * a, 
                        struct statistics_s const * b);


#endif
