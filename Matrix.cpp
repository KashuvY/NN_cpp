#pragma once
#include <vector>
#include <stdexcept>
#include <cmath>
#include "Vector.cpp"

class Matrix {
private:
    // Store data in a single contiguous array for better cache performance
    // Access element at row i, column j using: data_[i * cols_ + j]
    std::vector<float> data_;
    size_t rows_;
    size_t cols_;

public:
    Matrix(size_t rows, size_t cols) : data_(rows*cols), rows_(rows), cols_(cols) {}

    // Initialize weights using Xavier initialization
    void xavier_init() {
        // Xavier initialization helps prevent vanishing/exploding gradients
        // by keeping the variance of activations roughly constant across layers
        float limit = std::sqrt(6.0f / (rows_ * cols_));

        for (size_t i = 0; i < rows_ * cols_; ++i) {
            // Generate random number between -limit and limit
            data_[i] = (static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f) * limit;
        }
    }

    // Matrix-vector multiplication is our most performance-critical operation
    Vector multiply(const Vector& vec) const {
        if (vec.size() != cols_) {
            throw std::invalid_argument("Matrix and vector dimensions dont match for multiplication");
        }

        Vector result(rows_);
        for (size_t i = 0; i < rows_; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < cols_; ++j) {
                sum += data_[i * cols_ + j] * vec[j];
            }
            result[i] = sum;
        }
        return result;
    }

    /*
    Vector multiply_unrolled(const Vector& vec) const {
        Vector result(rows_);
        for (size_t i = 0; i < rows_; ++i) {
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
            
            // Process 4 elements at a time
            size_t j = 0;
            for (; j + 3 < cols_; j += 4) {
                sum0 += data_[i * cols_ + j] * vec[j];
                sum1 += data_[i * cols_ + j + 1] * vec[j + 1];
                sum2 += data_[i * cols_ + j + 2] * vec[j + 2];
                sum3 += data_[i * cols_ + j + 3] * vec[j + 3];
            }
            
            // Handle remaining elements
            float sum = sum0 + sum1 + sum2 + sum3;
            for (; j < cols_; ++j) {
                sum += data_[i * cols_ + j] * vec[j];
            }
            
            result[i] = sum;
        }
        return result;
    }

    Vector multiply_simd(const Vector& vec) const {
        Vector result(rows_);
        
        for (size_t i = 0; i < rows_; ++i) {
            // Initialize accumulator to zero
            float32x4_t sum_vec = vdupq_n_f32(0.0f);  // Creates vector of 4 zeros
            
            // Process 4 elements at a time using NEON
            size_t j = 0;
            for (; j + 3 < cols_; j += 4) {
                // Load 4 elements from matrix and vector
                float32x4_t a = vld1q_f32(&data_[i * cols_ + j]);
                float32x4_t b = vld1q_f32(&vec[j]);
                
                // Multiply and accumulate
                sum_vec = vmlaq_f32(sum_vec, a, b);  // sum_vec += a * b
            }
            
            // Sum the four elements of sum_vec
            float sum = vaddvq_f32(sum_vec);  // Horizontal add
            
            // Handle remaining elements
            for (; j < cols_; ++j) {
                sum += data_[i * cols_ + j] * vec[j];
            }
            
            result[i] = sum;
        }
        return result;
    }

    Vector multiply_blocked(const Vector& vec) const {
        Vector result(rows_);
        constexpr size_t BLOCK_SIZE = 64;  // Tune this based on your CPU's L1 cache size
        
        for (size_t i = 0; i < rows_; i += BLOCK_SIZE) {
            for (size_t j = 0; j < cols_; j += BLOCK_SIZE) {
                // Process a block of the matrix
                size_t i_end = std::min(i + BLOCK_SIZE, rows_);
                size_t j_end = std::min(j + BLOCK_SIZE, cols_);
                
                for (size_t ii = i; ii < i_end; ++ii) {
                    float sum = result[ii];  // Accumulate into existing sum
                    for (size_t jj = j; jj < j_end; ++jj) {
                        sum += data_[ii * cols_ + jj] * vec[jj];
                    }
                    result[ii] = sum;
                }
            }
        }
        return result;
    }

    Vector multiply_parallel(const Vector& vec) const {
        Vector result(rows_);
        
        #pragma omp parallel for
        for (size_t i = 0; i < rows_; ++i) {
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            size_t j = 0;
            
            for (; j + 3 < cols_; j += 4) {
                float32x4_t a = vld1q_f32(&data_[i * cols_ + j]);
                float32x4_t b = vld1q_f32(&vec[j]);
                sum_vec = vmlaq_f32(sum_vec, a, b);
            }
            
            float sum = vaddvq_f32(sum_vec);
            
            // Handle remaining elements
            for (; j < cols_; ++j) {
                sum += data_[i * cols_ + j] * vec[j];
            }
            
            result[i] = sum;
        }
        return result;
    }
    */

    float& at(size_t row, size_t col) {
        return data_[row * cols_ + col];
    }

    const float& at(size_t row, size_t col) const {
        return data_[row * cols_ + col];
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
};