#include <vector>

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