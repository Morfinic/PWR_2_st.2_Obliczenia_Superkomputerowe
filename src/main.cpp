#include <chrono>
#include <fstream>
#include "include/GeneticAlgorithm.h"
#include "include/Item.h"
#include "include/Knapsack.h"

void testParameterSensitivity(int numItems, int maxWeight, int maxValue, int capacity){
    Knapsack knapsack = Knapsack::generateRandom(numItems, maxWeight, maxValue, capacity);
    std::ofstream file("results/parameter_sensitivity.csv");
    file << "maxGenerations,populationSize,mutationRate,tournamentSize,avgFitness,avgTime[s]\n";
    std::vector<int> maxGenerations = {100, 200, 500};
    std::vector<int> populationSizes = {10, 50, 100};
    std::vector<double> mutationRates = {0.001, 0.01, 0.05};
    std::vector<int> tournamentSizes = {3, 5, 10};

    for (int maxGeneration: maxGenerations){
        for (int popSize : populationSizes) {
            for (double mutRate : mutationRates) {
                for (int tournamentSize : tournamentSizes) {
                    int runs = 10;
                    int totalFitness = 0;
                    double totalTime = 0;

                    for (int i = 0; i < runs; ++i) {
                        auto start = std::chrono::high_resolution_clock::now();
                        GeneticAlgorithm ga(knapsack, popSize, mutRate, tournamentSize, maxGeneration);
                        auto [best, _] = ga.solve(false, false);
                        auto end = std::chrono::high_resolution_clock::now();
                        totalFitness += ga.calculateFitness(best);
                        totalTime += std::chrono::duration<double>(end - start).count();
                    }
                    file << maxGeneration << "," << popSize << "," << mutRate << "," << tournamentSize << "," << (totalFitness / runs) << "," << (totalTime / runs) << "\n";
                }
            }
        }
    }
}

void testPerformance() {
    std::ofstream file("results/performance.csv");
    file << "items,fitness,time[s]\n";
    std::vector<int> sizes = {50, 100, 250, 500, 1000};

    for (int size : sizes) {
        Knapsack ks = Knapsack::generateRandom(size, 100, 200, size*12);
        GeneticAlgorithm ga(ks, 100, 0.01, 5, 500);

        auto start = std::chrono::high_resolution_clock::now();
        auto [best, _] = ga.solve(false, false);
        auto end = std::chrono::high_resolution_clock::now();
        
        file << size << "," << ga.calculateFitness(best) << "," << std::chrono::duration<double>(end - start).count() << "\n";
    }
}

void testStability(int numItems, int maxWeight, int maxValue, int capacity) {
    Knapsack knapsack = Knapsack::generateRandom(numItems, maxWeight, maxValue, capacity);
    std::ofstream file("results/stability.csv");
    file << "run,fitness\n";
    int runs = 30;

    std::vector<int> fitnesses;
    for (int i = 0; i < runs; ++i) {
        GeneticAlgorithm ga(knapsack);
        auto [best, _] = ga.solve(false, false);
        int fitness = ga.calculateFitness(best);
        fitnesses.push_back(fitness);
        file << i << "," << fitness << "\n";
    }

    double mean = std::accumulate(fitnesses.begin(), fitnesses.end(), 0.0) / runs;
    double sq_sum = std::inner_product(fitnesses.begin(), fitnesses.end(), fitnesses.begin(), 0.0);
    double stddev = std::sqrt(sq_sum / runs - mean * mean);

    file << "\nmeanFitness,stddev\n";
    file << mean << "," << stddev << "\n";
}

void defaultRun(int numItems, int maxWeight, int maxValue, int capacity) {
    Knapsack knapsack = Knapsack::generateRandom(numItems, maxWeight, maxValue, capacity);
    GeneticAlgorithm solver(knapsack);
    auto [bestSolution, lastGenSolution] = solver.solve(false, false);
    std::cout << "Best solution" << std::endl;
    knapsack.printSolution(bestSolution);
    std::cout << "Last gen solution" << std::endl;
    knapsack.printSolution(lastGenSolution);
}

int main (int argc, char* argv[]) {
    int numItems = 320;
    int maxWeight = 50;
    int maxValue = 100;
    int capacity = 4000;

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

    // testParameterSensitivity(numItems, maxWeight, maxValue, capacity);
    // testStability(numItems, maxWeight, maxValue, capacity);
    // testPerformance();

    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::high_resolution_clock::now();
    defaultRun(numItems, maxWeight, maxValue, capacity);
    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double T_Sekw = elapsed.count();
    std::cout << "Czas trwania algorytmu: " << T_Sekw/1000000 << "s\n";

    return 0;
}
