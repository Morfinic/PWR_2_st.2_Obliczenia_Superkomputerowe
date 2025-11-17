cmake_minimum_required(VERSION 3.30)
project(PWR_2_st.2_Obliczenia_Superkomputerowe)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_executable(
    ${PROJECT_NAME} src/main.cpp
        src/include/GeneticAlgorithm.h
        src/include/GeneticAlgorithm.cpp
        src/include/Item.h
        src/include/Item.cpp
        src/include/Knapsack.h
        src/include/Knapsack.cpp
        src/include/KnapsackGA.h
)
