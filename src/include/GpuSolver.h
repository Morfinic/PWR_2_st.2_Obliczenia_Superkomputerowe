#ifndef GPUSOLVER_H
#define GPUSOLVER_H

#include <vector>

struct GpuResult {
public:
    std::vector<int> bestChromosome;
    int bestFitness;
};

class GpuSolver {
public:
    static GpuResult solveKnapsackGPU(
        const std::vector<int>& weights,
        const std::vector<int>& values,
        int capacity,
        int populationSize,
        double mutationRate,
        int tournamentSize,
        int maxGenerations
    );
};

#endif //GPUSOLVER_H
