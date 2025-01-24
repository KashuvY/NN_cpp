#include <chrono>  // For timing operations
#include <iostream>
#include <random>  // For generating test data
#include <iomanip> // For formatting output
#include "layer.cpp"

// First, let's create a helper function to measure execution time
template<typename Func>
double measure_time(Func&& func) {
    // Get starting timepoint
    auto start = std::chrono::high_resolution_clock::now();
    
    // Execute the function
    func();
    
    // Get ending timepoint
    auto end = std::chrono::high_resolution_clock::now();
    
    // Calculate duration in microseconds
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    return duration.count() / 1000.0; // Convert to milliseconds
}

// Helper function to generate random test data
void fill_random(Vector& vec) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = dis(gen);
    }
}

int main() {
    // Test parameters
    const size_t MATRIX_SIZE = 10000;  // Size of square matrix
    const int NUM_TRIALS = 10;        // Number of trials for each method
    
    // Create test data
    Matrix matrix(MATRIX_SIZE, MATRIX_SIZE);
    Vector input_vector(MATRIX_SIZE);
    
    // Initialize with random data
    matrix.xavier_init();  // Use our existing initialization
    fill_random(input_vector);
    
    std::cout << "Testing matrix multiplication implementations...\n";
    std::cout << "Matrix size: " << MATRIX_SIZE << "x" << MATRIX_SIZE << "\n";
    std::cout << "Number of trials: " << NUM_TRIALS << "\n\n";
    
    // Test each implementation
    std::vector<std::pair<std::string, std::function<Vector()>>> implementations = {
        {"Basic", [&]() { return matrix.multiply(input_vector); }},
        {"Unrolled", [&]() { return matrix.multiply_unrolled(input_vector); }},
        {"SIMD", [&]() { return matrix.multiply_simd(input_vector); }},
        {"Blocked", [&]() { return matrix.multiply_blocked(input_vector); }},
        {"Parallel", [&]() { return matrix.multiply_parallel(input_vector); }}
    };
    
    // Run and time each implementation
    for (const auto& impl : implementations) {
        double total_time = 0.0;
        
        // Warm-up run (to avoid cold cache effects)
        impl.second();
        
        // Timed runs
        for (int i = 0; i < NUM_TRIALS; ++i) {
            double time = measure_time(impl.second);
            total_time += time;
        }
        
        double avg_time = total_time / NUM_TRIALS;
        std::cout << std::left << std::setw(15) << impl.first 
                  << ": " << std::fixed << std::setprecision(3) 
                  << avg_time << " ms\n";
    }
    
    return 0;
}