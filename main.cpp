#include <iostream>
#include "NeuralNetwork.cpp"

int main() {
    std::cout << "helo";
    // Create network with 2 inputs, 4 hidden neurons, and 1 output
    NeuralNetwork network({2, 4, 1});

    // Create training data
    std::vector<std::pair<Vector, Vector>> training_data;
    
    // Input vectors
    Vector input1(std::vector<float>{0, 0});
    Vector input2(std::vector<float>{0, 1});
    Vector input3(std::vector<float>{1, 0});
    Vector input4(std::vector<float>{1, 1});
    
    // Output vectors
    Vector output1(std::vector<float>{0});
    Vector output2(std::vector<float>{1});
    Vector output3(std::vector<float>{1});
    Vector output4(std::vector<float>{0});
    
    // Add training pairs
    training_data.push_back(std::make_pair(input1, output1));
    training_data.push_back(std::make_pair(input2, output2));
    training_data.push_back(std::make_pair(input3, output3));
    training_data.push_back(std::make_pair(input4, output4));

    // Training loop
    for (int epoch = 0; epoch < 1000; ++epoch) {
        float total_loss = 0.0f;
        
        // Since we're using C++14, replace structured bindings
        for (const auto& example : training_data) {
            const Vector& input = example.first;
            const Vector& target = example.second;
            total_loss += network.train(input, target);
        }

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " 
                      << total_loss / training_data.size() << std::endl;
        }
    }

    // Testing loop
    for (const auto& example : training_data) {
        const Vector& input = example.first;
        const Vector& target = example.second;
        Vector prediction = network.forward(input);
        std::cout << input[0] << " XOR " << input[1] << " = " 
                  << prediction[0] << " (expected " << target[0] << ")\n";
    }

    return 0;
}