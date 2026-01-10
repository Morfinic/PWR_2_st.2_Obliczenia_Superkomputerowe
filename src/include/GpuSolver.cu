#include "GpuSolver.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>

#define CHECK(call) \
{ \
const cudaError_t error = call; \
if (error != cudaSuccess) \
{ \
std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
<< cudaGetErrorString(error) << std::endl; \
exit(1); \
} \
}

__global__ void setupRngKernel(curandState* states, unsigned long seed, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void initPopulationKernel(int* population, curandState* states, int numItems, int popSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < popSize) {
        curandState localState = states[idx];
        for (int i = 0; i < numItems; i++) {
            population[idx * numItems + i] = (curand_uniform(&localState) > 0.5f) ? 1 : 0;
        }
        states[idx] = localState;
    }
}

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
        int offset = idx * numItems;

        for (int i = 0; i < numItems; i++) {
            if (population[offset + i] == 1) {
                currentWeight += weights[i];
                currentValue += values[i];
            }
        }

        if (currentWeight > capacity) {
            fitnessResults[idx] = 0;
        } else {
            fitnessResults[idx] = currentValue;
        }
    }
}

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

        int bestIdx1 = -1;
        int maxFit1 = -1;
        for(int t=0; t < tournamentSize; t++){
            int r = (int)(curand_uniform(&localState) * (popSize - 0.0001f));
            if(fitness[r] > maxFit1) {
                maxFit1 = fitness[r];
                bestIdx1 = r;
            }
        }

        int bestIdx2 = -1;
        int maxFit2 = -1;
        for(int t=0; t < tournamentSize; t++){
            int r = (int)(curand_uniform(&localState) * (popSize - 0.0001f));
            if(fitness[r] > maxFit2) {
                maxFit2 = fitness[r];
                bestIdx2 = r;
            }
        }

        int crossPoint = (int)(curand_uniform(&localState) * (numItems - 1));
        int myOffset = idx * numItems;
        int p1Offset = bestIdx1 * numItems;
        int p2Offset = bestIdx2 * numItems;

        for (int i = 0; i < numItems; i++) {
            int gene;
            if (i <= crossPoint)
                gene = oldPop[p1Offset + i];
            else
                gene = oldPop[p2Offset + i];

            if (curand_uniform(&localState) < mutationRate)
                gene = 1 - gene;

            newPop[myOffset + i] = gene;
        }

        states[idx] = localState;
    }
}

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
    size_t popSizeBytes = populationSize * numItems * sizeof(int);
    size_t itemsBytes = numItems * sizeof(int);

    int *d_popIn, *d_popOut, *d_weights, *d_values, *d_fitness;
    curandState *d_states;

    CHECK(cudaMalloc(&d_popIn, popSizeBytes));
    CHECK(cudaMalloc(&d_popOut, popSizeBytes));
    CHECK(cudaMalloc(&d_weights, itemsBytes));
    CHECK(cudaMalloc(&d_values, itemsBytes));
    CHECK(cudaMalloc(&d_fitness, populationSize * sizeof(int)));
    CHECK(cudaMalloc(&d_states, populationSize * sizeof(curandState)));

    CHECK(cudaMemcpy(d_weights, weights.data(), itemsBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_values, values.data(), itemsBytes, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocks = (populationSize + threadsPerBlock - 1) / threadsPerBlock;

    setupRngKernel<<<blocks, threadsPerBlock>>>(d_states, time(NULL), populationSize);
    initPopulationKernel<<<blocks, threadsPerBlock>>>(d_popIn, d_states, numItems, populationSize);
    CHECK(cudaDeviceSynchronize());

    GpuResult result;
    result.bestFitness = -1;
    std::vector<int> h_fitness(populationSize);

    for (int gen = 0; gen < maxGenerations; gen++) {
        calculateFitnessKernel<<<blocks, threadsPerBlock>>>(d_popIn, d_weights, d_values, d_fitness, numItems, capacity, populationSize);
        CHECK(cudaGetLastError());

        evolveKernel<<<blocks, threadsPerBlock>>>(d_popIn, d_popOut, d_fitness, d_states, numItems, populationSize, tournamentSize, mutationRate);
        CHECK(cudaGetLastError());

        std::swap(d_popIn, d_popOut);
    }

    CHECK(cudaDeviceSynchronize());

    calculateFitnessKernel<<<blocks, threadsPerBlock>>>(d_popIn, d_weights, d_values, d_fitness, numItems, capacity, populationSize);
    CHECK(cudaMemcpy(h_fitness.data(), d_fitness, populationSize * sizeof(int), cudaMemcpyDeviceToHost));

    int bestIdx = -1;
    for (int i = 0; i < populationSize; i++) {
        if (h_fitness[i] > result.bestFitness) {
            result.bestFitness = h_fitness[i];
            bestIdx = i;
        }
    }

    result.bestChromosome.resize(numItems);
    CHECK(cudaMemcpy(result.bestChromosome.data(), d_popIn + (bestIdx * numItems), itemsBytes, cudaMemcpyDeviceToHost));

    cudaFree(d_popIn); cudaFree(d_popOut); cudaFree(d_weights);
    cudaFree(d_values); cudaFree(d_fitness); cudaFree(d_states);

    return result;
}
