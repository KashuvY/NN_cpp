#pragma once
#include <vector>
#include <stdexcept>
#include <cmath>
#include "Vector.cpp"
#include "Matrix.cpp"
#include "Activation.cpp"

// Handles the mathematical operations and storage of layer weights
class WeightMatrix {
private:
    // We'll decide on the exact storage mechanism later
    Matrix weights_;
    Vector bias_;
    
public:
    WeightMatrix(size_t input_size, size_t output_size) 
        : weights_(output_size, input_size), bias_(output_size)  // initialize matrix and bias vector
    {
        // initialize weights using xavier initialization
        weights_.xavier_init();
        bias_.uniform_init();
    }
    
    // Forward pass: output = weights * input + bias
    Vector multiply(const Vector& input) const {
        Vector output = weights_.multiply(input);
        for (size_t i = 0; i < bias_.size(); ++i) {
            output[i] += bias_[i];
        }

        return output;
    }

    // update weights and biases during backpropogation
    void update(const Matrix& weight_gradients, const Vector& bias_gradients, float learning_rate = 0.01f) {
        // verify dimensions match
        if (weight_gradients.rows() != weights_.rows() || 
            weight_gradients.cols() != weights_.cols()) {
                throw std:: invalid_argument("Weight gradient dimensions don't match");
            } 
        
        if (bias_gradients.size() != bias_.size()) {
            throw std::invalid_argument("Bias gradient dimensions don't match");
        }

        for (size_t i = 0; i < weights_.rows(); ++i) {
            for (size_t j = 0; j < weights_.cols(); ++i) {
                weights_.at(i, j) -= learning_rate * weight_gradients.at(i, j);
            }
        }

        for (size_t i = 0; i < bias_.size(); ++i) {
            bias_[i] -= learning_rate * bias_gradients[i];
        }
    }
    
    // Getter methods for accessing dimensions
    size_t input_size() const { return weights_.cols(); }
    size_t output_size() const { return weights_.rows(); }

};