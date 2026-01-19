#include "GpuSolver.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>

// Sprawdzanie błedów CUDA - pożyczone
// Jeśli funkcja zwróci błąd, program zostanie przerwany i wypisze błąd
#define CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Inicjalizacja stanu generatora liczb losowych dla każdego wątka
__global__ void setupRngKernel(curandState* states, unsigned long seed, int N) {
    // Szukanie id wątku
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Sprawdzenie bounds
    if (idx < N) {
        // Inicjalizacja stanu generatora dla wątka 'idx'
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Inicjalizowanie populacji, losowe rozwiązania (0 lub 1) dla każdego osobnika
__global__ void initPopulationKernel(int* population, curandState* states, int numItems, int popSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < popSize) {
        // Pobranie stanu generatora
        curandState localState = states[idx];
        for (int i = 0; i < numItems; i++) {
            // curand_uniform zwraca <0.0; 1.0>, więc 0.5 to 50% szansy na 1
            population[idx * numItems + i] = (curand_uniform(&localState) > 0.5f) ? 1 : 0;
        }
        // Zapis zaktualizowanego stanu generatora
        states[idx] = localState;
    }
}

// Obliczanie fitness
__global__ void calculateFitnessKernel(
    const int* population,
    const int* weights,
    const int* values,
    int* fitnessResults,
    int numItems,
    int capacity,
    int popSize
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < popSize) {
        int currentWeight = 0;
        int currentValue = 0;

        // Obliczanie przesunięcia w pamięci, gdzie zaczyna się osobnik
        int offset = idx * numItems;

        // Sumowanie wag i wartości przedmiotów
        for (int i = 0; i < numItems; i++) {
            if (population[offset + i] == 1) {
                currentWeight += weights[i];
                currentValue += values[i];
            }
        }

        // Sprawdzenie warunku plecaka (czy nie dźwiga za dużo)
        if (currentWeight > capacity) {
            fitnessResults[idx] = 0;
        } else {
            fitnessResults[idx] = currentValue;
        }
    }
}

// Ewolucja (selekcja + krzyżowanie + mutacja)
// Każdy wątek tworzy jednego nowego osobnika
__global__ void evolveKernel(
    const int* oldPop,
    int* newPop,
    const int* fitness,
    curandState* states,
    int numItems,
    int popSize,
    int tournamentSize,
    float mutationRate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < popSize) {
        curandState localState = states[idx];

        // Selekcja turniejowa (rodzic 1)
        int bestIdx1 = -1;
        int maxFit1 = -1;

        // Losowanie kilku kandytatów i wybór najlepszego
        for(int t=0; t < tournamentSize; t++){
            int r = (int)(curand_uniform(&localState) * (popSize - 0.0001f));
            if(fitness[r] > maxFit1) {
                maxFit1 = fitness[r];
                bestIdx1 = r;
            }
        }

        // Selekcja turniejowa (rodzic 2)
        int bestIdx2 = -1;
        int maxFit2 = -1;
        for(int t=0; t < tournamentSize; t++){
            int r = (int)(curand_uniform(&localState) * (popSize - 0.0001f));
            if(fitness[r] > maxFit2) {
                maxFit2 = fitness[r];
                bestIdx2 = r;
            }
        }

        // Krzyżowanie
        // Wyznaczamy punkt przecięcia: geny przed od rodzica1, a po od rodzica 2
        int crossPoint = (int)(curand_uniform(&localState) * (numItems - 1));

        // Offsety w pamięci dla rodziców i dziecka
        int myOffset = idx * numItems;
        int p1Offset = bestIdx1 * numItems;
        int p2Offset = bestIdx2 * numItems;

        // Kopiowanie genów
        for (int i = 0; i < numItems; i++) {
            int gene; // Czy przedmiot zostanie wzięty do plecaka
            if (i <= crossPoint)
                gene = oldPop[p1Offset + i];
            else
                gene = oldPop[p2Offset + i];

            // Mała szansa na mutację genu
            if (curand_uniform(&localState) < mutationRate)
                gene = 1 - gene;

            // Zapis do nowej populacji
            newPop[myOffset + i] = gene;
        }

        // Zapis stanu generatora
        states[idx] = localState;
    }
}

// Funkcja host (CPU)
GpuResult GpuSolver::solveKnapsackGPU(
    const std::vector<int>& weights,
    const std::vector<int>& values,
    int capacity,
    int populationSize,
    double mutationRate,
    int tournamentSize,
    int maxGenerations
) {
    int numItems = weights.size();

    // Obliczany rozmiar pamięci potrzebny do zaalokowania dla bufora
    size_t popSizeBytes = populationSize * numItems * sizeof(int);
    size_t itemsBytes = numItems * sizeof(int);

    // Wskaźniki na pamięć GPU
    int *d_popIn, *d_popOut, *d_weights, *d_values, *d_fitness;
    curandState *d_states;

    // Alokacja pamięci na GPU
    CHECK(cudaMalloc(&d_popIn, popSizeBytes));
    CHECK(cudaMalloc(&d_popOut, popSizeBytes));
    CHECK(cudaMalloc(&d_weights, itemsBytes));
    CHECK(cudaMalloc(&d_values, itemsBytes));
    CHECK(cudaMalloc(&d_fitness, populationSize * sizeof(int)));
    CHECK(cudaMalloc(&d_states, populationSize * sizeof(curandState)));

    // Kopiowanie danych z CPU na GPU
    CHECK(cudaMemcpy(d_weights, weights.data(), itemsBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_values, values.data(), itemsBytes, cudaMemcpyHostToDevice));

    // Konfiguracja siatki wątków
    int threadsPerBlock = 64;
    int blocks = (populationSize + threadsPerBlock - 1) / threadsPerBlock;

    // Inicjalizacja ziarna i generatorów
    setupRngKernel<<<blocks, threadsPerBlock>>>(d_states, time(NULL), populationSize);
    // Stworzenie pierwszej populacji
    initPopulationKernel<<<blocks, threadsPerBlock>>>(d_popIn, d_states, numItems, populationSize);
    CHECK(cudaDeviceSynchronize());

    GpuResult result;
    result.bestFitness = -1;
    // Bufor CPU na wynik
    std::vector<int> h_fitness(populationSize);

    // Główna pętla ewolucyjna
    for (int gen = 0; gen < maxGenerations; gen++) {
        // 1. Policz fitness dla populacji (d_popIn)
        calculateFitnessKernel<<<blocks, threadsPerBlock>>>(d_popIn, d_weights, d_values, d_fitness, numItems, capacity, populationSize);
        CHECK(cudaGetLastError());

        // 2. Stwórz populację d_popOut na podstawie d_popIn
        evolveKernel<<<blocks, threadsPerBlock>>>(d_popIn, d_popOut, d_fitness, d_states, numItems, populationSize, tournamentSize, mutationRate);
        CHECK(cudaGetLastError());

        // 3. Zamiast kopiowania zamiana wskaźników
        std::swap(d_popIn, d_popOut);
    }

    CHECK(cudaDeviceSynchronize());

    // Liczenie fitness ostatecznej populacji
    calculateFitnessKernel<<<blocks, threadsPerBlock>>>(d_popIn, d_weights, d_values, d_fitness, numItems, capacity, populationSize);
    // Kopiowanie fitness na CPU
    CHECK(cudaMemcpy(h_fitness.data(), d_fitness, populationSize * sizeof(int), cudaMemcpyDeviceToHost));

    // Szukanie najlepszego wyniku
    int bestIdx = -1;
    for (int i = 0; i < populationSize; i++) {
        if (h_fitness[i] > result.bestFitness) {
            result.bestFitness = h_fitness[i];
            bestIdx = i;
        }
    }

    // Kopiowanie najlepszego wyniku z GPU
    result.bestChromosome.resize(numItems);
    CHECK(cudaMemcpy(result.bestChromosome.data(), d_popIn + (bestIdx * numItems), itemsBytes, cudaMemcpyDeviceToHost));

    // Czyszczenie pamięci
    cudaFree(d_popIn); cudaFree(d_popOut); cudaFree(d_weights);
    cudaFree(d_values); cudaFree(d_fitness); cudaFree(d_states);

    return result;
}
