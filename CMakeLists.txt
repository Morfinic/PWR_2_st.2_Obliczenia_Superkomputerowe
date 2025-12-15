cmake_minimum_required(VERSION 3.28.1)
project(PWR_2_st.2_Obliczenia_Superkomputerowe)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE)
    find_program(BREW_PROG brew)
    if(BREW_PROG)
        execute_process(COMMAND ${BREW_PROG} --prefix libomp OUTPUT_VARIABLE LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
    else()
        set(LIBOM_PREFIX "/opt/homebrew/opt/libomb")
    endif()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Xpreprocessor -fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")

    include_directories(${LIBOMP_PREFIX}/include)
    link_directories(${LIBOMP_PREFIX}/lib)

    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_C_LIB_NAMES "omp")

    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")

    set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
endif()

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

find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()
