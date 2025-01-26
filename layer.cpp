#include <vector>
#include <stdexcept>
#include <cmath>
#include <arm_neon.h>  // For NEON instrinsics
#include "/opt/homebrew/opt/libomp/include/omp.h"


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
    Vector forward(const Vector& input) const override;
    Vector backward(const Vector& input, const Vector& gradient) const override;
};

// Handles the mathematical operations and storage of layer weights
class WeightMatrix {
private:
    // We'll decide on the exact storage mechanism later
    Matrix weights_;
    Vector bias_;
    
public:
    WeightMatrix(size_t input_size, size_t output_size);
    
    // Forward declaration of core operations
    Vector multiply(const Vector& input) const;
    void update(const Matrix& weight_gradients, const Vector& bias_gradients);
};

// A Layer combines weights with an activation function
class Layer {
private:
    WeightMatrix weights_;
    std::unique_ptr<ActivationFunction> activation_;
    
    // Pre-allocated buffers for intermediate results
    mutable Vector pre_activation_buffer_;  // Stored for backprop
    mutable Vector activation_buffer_;
    
public:
    Layer(size_t input_size, size_t output_size, 
          std::unique_ptr<ActivationFunction> activation);
    
    // Core operations
    const Vector& forward(const Vector& input) const;
    Vector backward(const Vector& gradient) const;
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