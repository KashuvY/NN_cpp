cmake_minimum_required(VERSION 3.10)
project(NeuralNetwork)

# Enable C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Enable OpenMP if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Add executable
add_executable(neural_network main.cpp)

# Link OpenMP if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(neural_network PUBLIC OpenMP::OpenMP_CXX)
endif()