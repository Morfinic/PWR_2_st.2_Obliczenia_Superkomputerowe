#include <chrono>
#include <fstream>
#include "include/GeneticAlgorithm.h"
#include "include/Item.h"
#include "include/Knapsack.h"
#include <omp.h>

void defaultRun(int numItems, int maxWeight, int maxValue, int capacity, bool isParallel) {
    Knapsack knapsack = Knapsack::generateRandom(numItems, maxWeight, maxValue, capacity);
    GeneticAlgorithm solver(knapsack);
    auto [bestSolution, lastGenSolution] = solver.solve(false, false, isParallel);
    std::cout << "Best solution" << std::endl;
    knapsack.printSolution(bestSolution);
}

int main (int argc, char* argv[]) {
    int numItems = 50;
    int maxWeight = 50;
    int maxValue = 100;
    int capacity = 1000;

    std::vector<Item> items;
    if(argc > 1){
        numItems = std::stoi(argv[1]);
    }
    if(argc > 2){
        maxWeight = std::stoi(argv[2]);
    }
    if(argc > 3){
        maxValue = std::stoi(argv[3]);
    }
    if(argc > 4){
        capacity = std::stoi(argv[4]);
    }

    // Ustawianie dostepnych threads dla OpenMP
    int num_of_threads = omp_get_max_threads();
    std::cout<< "Liczba rdzeni logicznych: " << num_of_threads << "\n\n";
    omp_set_num_threads(num_of_threads);


    auto start = std::chrono::high_resolution_clock::now();
    // Uruchamienia algorytmu
    defaultRun(numItems, maxWeight, maxValue, capacity, true);
    // ----------------------
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double T_Sekw = elapsed.count();
    std::cout << "Czas trwania algorytmu: " << T_Sekw/1000000 << "s\n";

    return 0;
}
