#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream> // Do zapisu plików
#include <omp.h>
#include <c++/12/vector>

#include "include/GeneticAlgorithm.h"
#include "include/GpuSolver.h"
#include "include/Knapsack.h"

// =============================================================
// KONFIGURACJA EKSPERYMENTÓW
// =============================================================
const int NUM_TRIALS = 100;

// =============================================================
// FUNKCJA POMIAROWA
// =============================================================
double measureSingleRun(int numItems, int capacity, bool isParallel) {
    // 1. Parametry problemu
    int maxWeight = 50;
    int maxValue = 100;
    
    // 2. PARAMETRY ALGORYTMU
    int popSize = 1000;       
    double mutRate = 0.01;
    int tourSize = 5;
    int generations = 500;    

    // 3. Generowanie instancji
    Knapsack knapsack = Knapsack::generateRandom(numItems, maxWeight, maxValue, capacity);
    
    // 4. Inicjalizacja solvera
    GeneticAlgorithm solver(knapsack, popSize, mutRate, tourSize, generations);

    // 5. Pomiar czasu
    auto start = std::chrono::high_resolution_clock::now();
    
    // verbose=false, log=false
    solver.solve(false, false, isParallel);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    return elapsed.count(); // Czas w sekundach
}

double measureSingleCudaRun(int numItems, int capacity) {
    // 1. Parametry problemu
    int maxWeight = 50;
    int maxValue = 100;

    // 2. PARAMETRY ALGORYTMU
    int popSize = 1000;
    double mutRate = 0.01;
    int tourSize = 5;
    int generations = 500;

    Knapsack knapsack = Knapsack::generateRandom(numItems, maxWeight, maxValue, capacity);

    std::vector<int> weights;
    std::vector<int> values;

    const auto &items = knapsack.getItems();
    for (const auto &item : items) {
        weights.push_back(item.weight);
        values.push_back(item.value);
    }

    auto start = std::chrono::high_resolution_clock::now();

    GpuResult res = GpuSolver::solveKnapsackGPU(
        weights,
        values,
        capacity,
        popSize,
        mutRate,
        tourSize,
        generations
    );

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    return elapsed.count(); // Czas w sekundach
}

// =============================================================
// EKSPERYMENT 1: Skalowalność sekwencyjna (Zależność od N)
// =============================================================
void runSequentialScalabilityExp() {
    std::cout << "=== EKSPERYMENT 1: Skalowalnosc Sekwencyjna ===" << std::endl;
    
    // Otwarcie pliku CSV
    std::ofstream csvFile("wyniki_sekwencyjne.csv");
    csvFile << "N,Capacity,AvgTime_s\n"; // Nagłówek CSV

    std::vector<int> itemSizes = {200, 500, 1000, 2000, 5000};

    // Tylko 1 wątek
    omp_set_num_threads(1);

    for (int n : itemSizes) {
        int capacity = n * 5; 
        
        std::cout << "Testowanie N=" << n << " (Wait...)" << std::flush;
        
        double totalTime = 0.0;
        for (int i = 0; i < NUM_TRIALS; i++) {
            totalTime += measureSingleRun(n, capacity, false);
            // Pasek postępu co 10 prób
            if (i % 20 == 0) std::cout << "." << std::flush;
        }
        double avgTime = totalTime / NUM_TRIALS;

        // Wypisanie na ekran
        std::cout << " Done. Sredni czas: " << avgTime << " s" << std::endl;

        // Zapis do pliku
        csvFile << n << "," << capacity << "," << avgTime << "\n";
    }
    csvFile.close();
    std::cout << "Wyniki zapisano do: wyniki_sekwencyjne.csv\n\n";
}

// =============================================================
// EKSPERYMENT 2: Przyspieszenie (Speedup)
// =============================================================
void runParallelScalingExp() {
    std::cout << "=== EKSPERYMENT 2: Skalowalnosc Rownolegla ===" << std::endl;
    
    std::ofstream csvFile("wyniki_rownolegle.csv");
    csvFile << "Threads,AvgTime_s,Speedup\n"; // Nagłówek

    int n = 3000; // Stały duży problem
    int capacity = n * 5;
    
    std::cout << "Problem: N=" << n << std::endl;

    int maxThreads = omp_get_max_threads();
    std::vector<int> threadsToCheck = {1, 2, 4, 8};
    
    bool maxAdded = false;
    for(int t : threadsToCheck) if(t == maxThreads) maxAdded = true;
    if (!maxAdded && maxThreads > 1) threadsToCheck.push_back(maxThreads);

    double timeSequential = 0.0;

    for (int t : threadsToCheck) {
        omp_set_num_threads(t);
        
        std::cout << "Testowanie watkow: " << t << " (Wait...)" << std::flush;

        double totalTime = 0.0;
        for (int i = 0; i < NUM_TRIALS; i++) {
            totalTime += measureSingleRun(n, capacity, true);
             if (i % 20 == 0) std::cout << "." << std::flush;
        }
        double avgTime = totalTime / NUM_TRIALS;

        if (t == 1) {
            timeSequential = avgTime;
        }
        
        double speedup = timeSequential / avgTime;

        std::cout << " Done. Czas: " << avgTime << " s | Speedup: " << speedup << "x" << std::endl;

        // Zapis do pliku
        csvFile << t << "," << avgTime << "," << speedup << "\n";
    }
    csvFile.close();
    std::cout << "Wyniki zapisano do: wyniki_rownolegle.csv\n\n";
}

// =============================================================
// EKSPERYMENT 2: CUDA (Zależność od N)
// =============================================================
void runCudaExperiment() {
    std::cout << "=== EKSPERYMENT 3: CUDA ===" << std::endl;

    // Otwarcie pliku CSV
    std::ofstream csvFile("wyniki_Cuda.csv");
    csvFile << "N,Capacity,AvgTime_s\n"; // Nagłówek CSV

    std::vector<int> itemSizes = {200, 500, 1000, 2000, 5000};

    // Tylko 1 wątek
    omp_set_num_threads(1);

    for (int n : itemSizes) {
        int capacity = n * 5;

        std::cout << "Testowanie N=" << n << " (Wait...)" << std::flush;

        double totalTime = 0.0;
        for (int i = 0; i < NUM_TRIALS; i++) {
            totalTime += measureSingleCudaRun(n, capacity);
            // Pasek postępu co 10 prób
            if (i % 20 == 0) std::cout << "." << std::flush;
        }
        double avgTime = totalTime / NUM_TRIALS;

        // Wypisanie na ekran
        std::cout << " Done. Sredni czas: " << avgTime << " s" << std::endl;

        // Zapis do pliku
        csvFile << n << "," << capacity << "," << avgTime << "\n";
    }
    csvFile.close();
    std::cout << "Wyniki zapisano do: wyniki_cuda.csv\n\n";
}

int main(int argc, char* argv[]) {
    // runSequentialScalabilityExp();
    // runParallelScalingExp();
    runCudaExperiment();

    return 0;
}
