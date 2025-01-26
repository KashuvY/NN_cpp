#pragma once
#include <vector>
#include <stdexcept>
#include <cmath>
#include "Vector.cpp"
#include "Matrix.cpp"
#include "Activation.cpp"
#include "WeightMatrix.cpp"

class Layer {
private:
    // The weight matrix handles the linear transformation (Wx + b)
    WeightMatrix weights_;

    // Smart pointer to activation function - allows polymorphic behavior
    // We use unique_ptr because each layer owns its activation function
    std::unique_ptr<ActivationFunction> activation_;

    // These buffers store intermediate results during forward pass
    // They're marked mutable because they're cached values that don't
    // affect the logical state of the layer
    // preinitialized during compilation to save time during runtime
    mutable Vector forward_input_buffer_;   // Stores input during forward pass
    mutable Vector pre_activation_buffer_;  // Stores Wx + b
    mutable Vector activation_buffer_;      // Stores activation(Wx + b)
    mutable Vector activation_gradient_bugger_; // Stores gradient through activation

public:
    // Constructor initializes a layer with specific dimensions and activation
    Layer(size_t input_size, size_t output_size,
         std::unique_ptr<ActivationFunction> activation)
        : weights_(input_size, output_size),
          activation_(std::move(activation)),
          forward_input_buffer_(input_size),
          pre_activation_buffer_(output_size),
          activation_buffer_(output_size),
          activation_gradient_bugger_(output_size) {}

    // Forward pass: transforms input through weights and activation
    // Returns reference to avoid copying, const because it doesn't modify layer
    const Vector& forward(const Vector& input) const {
        // 1. Apply weight matrix to input (store in pre_activation_buffer_)
        pre_activation_buffer_ = weights_.multiply(input);
        // 2. Apply activation function (store in activation_buffer_)
        activation_buffer_ = activation_->forward(pre_activation_buffer_);
        // 3. Return activation_buffer_
        return activation_buffer_;
    }

    // Backward pass: computes gradients for backpropagation
    // Takes gradient from next layer, returns gradient for previous layer
    Vector backward(const Vector& gradient) const {
        // 1. Compute activation function gradient using stored pre_activation
        Vector activation_gradient = activation_->backward(pre_activation_buffer_, gradient);
        // 2. Compute weight gradients
        // 3. Compute and return gradient for previous layer
    }

private:
    // Helper function to compute weight gradients
    Matrix compute_weight_gradients() const {
        // 1. Create gradient matrix of proper size
        // 2. Compute outer product of activation_gradient_ and forward_input_
        // 3. Return weight gradients
    }

    // Helper function to compute input gradients for previous layer
    Vector compute_input_gradients() const {
        // 1. Create input gradient vector of proper size
        // 2. Compute matrix-vector product of transposed weights and activation_gradient_
        // 3. Return input gradients
    }
};






// The neural network composes layers together
class NeuralNetwork {
private:
    std::vector<Layer> layers_;
    
public:
    // Builder pattern for constructing the network
    class Builder;
    
    Vector forward(const Vector& input) const;
    void backward(const Vector& expected_output);
    void update_weights(float learning_rate);
};