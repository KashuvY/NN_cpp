#pragma once
#include <vector>
#include <iostream>
#include "Layer.cpp"

// The neural network composes layers together
class NeuralNetwork {
private:
    std::vector<Layer> layers_;
    float learning_rate_;

    // Store intermediate values during training
    mutable std::vector<Vector> layer_outputs_;
    
public:
    // Constructor takes a vector of layer sizes and creates the network
    NeuralNetwork(const std::vector<size_t>& layer_sizes, float learning_rate = 0.01f) 
        : learning_rate_(learning_rate)
    {
        // Need at least two layers (input and output)
        if (layer_sizes.size() < 2) {
            throw std::invalid_argument("Network must have at least input and output layers");
        }

        // Create each layer
        for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
            std::unique_ptr<ActivationFunction> activation;
            if (i == layer_sizes.size() - 2) {
                std::cout << "Creating output layer with Sigmoid activation" << std::endl;
                activation = std::make_unique<Sigmoid>();
            } else {
                std::cout << "Creating hidden layer with ReLU activation" << std::endl;
                activation = std::make_unique<ReLU>();
            }

            std::cout << "Layer " << i << ": " << layer_sizes[i] << " -> " 
                      << layer_sizes[i+1] << " neurons" << std::endl;

            layers_.emplace_back(layer_sizes[i], layer_sizes[i+1], std::move(activation));
        }

        // Initialize storage for layer outputs
        layer_outputs_.resize(layer_sizes.size());
    }

    Vector forward(const Vector& input) const {
        layer_outputs_[0] = input;
        for (size_t i = 0; i < layers_.size(); ++i) {
            layer_outputs_[i + 1] = layers_[i].forward(layer_outputs_[i]);
        }
        return layer_outputs_.back(); // final layer output
    }

    float train(const Vector& input, const Vector& target) {
        Vector prediction = forward(input);
        float loss = compute_loss(prediction, target);
        Vector gradient = compute_loss_gradient(prediction, target);
        for (int i = layers_.size() - 1; i >= 0; --i) {
            gradient = layers_[i].backward(gradient);
        }
        return loss;
    }

private:
    // Binary cross-entropy loss
    float compute_loss(const Vector& prediction, const Vector& target) const {
        float loss = 0.0f;
        for (size_t i = 0; i < prediction.size(); ++i) {
            // avoid log(0) by adding small epsilon
            float p = std::max(std::min(prediction[i], 1.0f - 1e-7f), 1e-7f);
            loss += -(target[i] * std::log(p) + (1 - target[i]) * std::log(1 - p));
        }
        return loss;
    }
    
    Vector compute_loss_gradient(const Vector& prediction, const Vector& target) const {
        Vector gradient(prediction.size());
        for (size_t i = 0; i < prediction.size(); ++i) {
            // Derivative of binary cross-entropy
            gradient[i] = (prediction[i] - target[i]) / (prediction[i] * (1 - prediction[i]));
        }
        return gradient;
    }

};