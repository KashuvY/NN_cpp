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
    mutable Vector activation_gradient_buffer_; // Stores gradient through activation

public:
    // Constructor initializes a layer with specific dimensions and activation
    Layer(size_t input_size, size_t output_size,
         std::unique_ptr<ActivationFunction> activation)
        : weights_(input_size, output_size),
          activation_(std::move(activation)),
          forward_input_buffer_(input_size),
          pre_activation_buffer_(output_size),
          activation_buffer_(output_size),
          activation_gradient_buffer_(output_size) {}

    // Forward pass: transforms input through weights and activation
    // Returns reference to avoid copying, const because it doesn't modify layer
    const Vector& forward(const Vector& input) const {
        // 1. Apply weight matrix to input (store in pre_activation_buffer_)
        forward_input_buffer_ = input;
        pre_activation_buffer_ = weights_.multiply(forward_input_buffer_);
        // 2. Apply activation function (store in activation_buffer_)
        activation_buffer_ = activation_->forward(pre_activation_buffer_);
        // 3. Return activation_buffer_
        return activation_buffer_;
    }

    // Backward pass: computes gradients for backpropagation
    // Takes gradient from next layer, returns gradient for previous layer
    Vector backward(const Vector& gradient) {
        // 1. Compute activation function gradient using stored pre_activation
        // This tells us how changes in our pre-activation values affect the loss
        activation_gradient_buffer_ = activation_->backward(pre_activation_buffer_, gradient);
        
        // 2. Compute weight gradients using our helper function
        // This tells us how to adjust each weight to reduce the error
        Matrix weight_gradients = compute_weight_gradients();
        
        // 3. Update weights using the computed gradients
        // Here we need to update both weights and biases to improve our predictions
        weights_.update(weight_gradients, gradient);  // gradient is used for bias updates
        
        // 4. Compute and return gradient for previous layer
        // This tells the previous layer how it should change its outputs
        return compute_input_gradients();
    }

private:
    Matrix compute_weight_gradients() const {
        Matrix gradients(weights_.output_size(), weights_.input_size());
        for (size_t i = 0; i < weights_.output_size(); ++i) {
            for (size_t j = 0; j < weights_.input_size(); ++j) {
                gradients.at(i, j) = activation_gradient_buffer_[i] * forward_input_buffer_[j];
            }
        }
        return gradients;
    }

    Vector compute_input_gradients() const {
        Vector input_grad(weights_.input_size());
        for (size_t i = 0; i < weights_.input_size(); ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < weights_.output_size(); ++j) {
                sum += weights_.at(j, i) * activation_gradient_buffer_[j];
            }
            input_grad[i] = sum;
        }
        return input_grad;
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