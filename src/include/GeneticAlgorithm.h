#pragma once

#include <vector>
#include <algorithm>
#include <random>
#include <fstream>
#include "Item.h"
#include "Knapsack.h"
#include "KnapsackGA.h"

class GeneticAlgorithm{
    private:
        Knapsack& knapsack;
        int populationSize;
        int chromosomeLength;
        double mutationRate;
        int tournamentSize;
        int maxGenerations;
        std::mt19937 generator;

        std::vector<KnapsackGA::Solution> generateInitialPopulation();
        KnapsackGA::Solution tournamentSelection(std::vector<KnapsackGA::Solution>& population);
        std::pair<KnapsackGA::Solution, KnapsackGA::Solution> crossover(KnapsackGA::Solution& parent1, KnapsackGA::Solution& parent2);
        void mutate(KnapsackGA::Solution& chromosome);
        
        public:
        GeneticAlgorithm(
            Knapsack& knapsack,
            int populationSize = 100,
            double mutationRate = 0.01,
            int tournamentSize = 5,
            int maxGenerations = 200
        );
        std::pair<KnapsackGA::Solution, KnapsackGA::Solution> solve(bool verbose=true, bool log=false);
        int calculateFitness(KnapsackGA::Solution& chromosome);
};