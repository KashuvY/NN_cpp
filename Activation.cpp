#pragma once
#include <vector>
#include <stdexcept>
#include <cmath>
#include "Vector.cpp"
#include "Matrix.cpp"


// Abstract base class for activation functions
class ActivationFunction {
public:
    virtual ~ActivationFunction() = default;
    
    // Pure virtual functions that derived classes must implement
    virtual Vector forward(const Vector& input) const = 0;
    virtual Vector backward(const Vector& input, const Vector& gradient) const = 0;
};

// Concrete activation functions inherit from base class
class ReLU : public ActivationFunction {
public:
    Vector forward(const Vector& input) const override {
        // Create vector to store results
        Vector result(input.size());

        for (size_t i = 0; i < input.size(); ++i){
            result[i] = std::max(0.0f, input[i]);
        }

        return result;
    }
    Vector backward(const Vector& input, const Vector& gradient) const override {
        Vector result(input.size());

        // For numerical stability, we will say the derivative at 0 is 0
        for (size_t i = 0; i < input.size(); ++i){
            result[i] = input[i] > 0.0f ? gradient[i] : 0.0f;
        }

        return result;
    }
};

class Sigmoid : public ActivationFunction {
public:
    Vector forward(const Vector& input) const override {
        // Create vector to store results
        Vector result(input.size());

        // split into +/- components for numerical stability
        for (size_t i = 0; i < input.size(); ++i){
            if (input[i] < 0.0f) {
                result[i] = std::exp(input[i]) / (1 + std::exp(input[i]));
            }
            else result[i] = 1 / (1 + std::exp(-input[i]));
        }

        return result;
    }
    Vector backward(const Vector& input, const Vector& gradient) const override {
        Vector result(input.size());
        Vector sigmoid_x = forward(input);

        // s'(x) = s(x) * (1-s(x))
        for (size_t i = 0; i < input.size(); ++i){
            result[i] = sigmoid_x[i] * (1.0f - sigmoid_x[i]) * gradient[i];
        }

        return result;
    }
};