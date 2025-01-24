#include <vector>
#include <stdexcept>
#include <cmath>

class Vector {
private:
   // We store our data in a contiguous block of memory for cache efficiency
   std::vector<float> data_;

public:
    // Constructor that creates a vector of gizen size
    explicit Vector(size_t size) : data_(size, 0.0f) {};
    
    // Constructor that creates a vector from existing data
    Vector(const std::vector<float>& data) : data_(data) {}

    // Copy constructor
    Vector(const Vector& other) : data_(other.data_) {}

    // Assignment operator
    Vector& operator=(const Vector& other) {
        if (this != &other) {
            data_ = other.data_;
        }
        return *this;
    }

    // Access elements (both const and non-const version)
    float& operator[](size_t index) {
        return data_[index];
    }

    const float& operator[](size_t index) const {
        return data_[index];
    }

    // Basic vector operations we will need
    Vector& operator+=(const Vector& other){
        if (other.size() != size()) {
            throw std::invalid_argument("Vector dimensions don't match for addition");
        }

        for (size_t i = 0; i < size(); ++i){
            data_[i] += other[i];
        }
        return *this;
    }

    // Element-wise multiplication (Hadamard product) - will be needed for backpropogation
    Vector hadamard(const Vector& other) const {
        if (other.size() != size()) {
            throw std::invalid_argument("Vector dimensions don't match for Hadamard product");
        }

        Vector result(size());
        for (size_t i = 0; i < size(); ++i){
            result[i] = data_[i] * other[i];
        }
        return result;
    }

    // Size accessor
    size_t size() const {
        return data_.size();
    }

};

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

    float& at(size_t row, size_t col) {
        return data_[row * cols_ + col];
    }

    const float& at(size_t row, size_t col) const {
        return data_[row * cols_ + col];
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
};

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