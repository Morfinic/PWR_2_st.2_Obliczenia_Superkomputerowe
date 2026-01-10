#include "GeneticAlgorithm.h"
#include <omp.h>

GeneticAlgorithm::GeneticAlgorithm(
    Knapsack& knapsack,
    int populationSize,
    double mutationRate,
    int tournamentSize,
    int maxGenerations
) :
    knapsack(knapsack),
    populationSize(populationSize),
    chromosomeLength(knapsack.getItemCount()),
    mutationRate(mutationRate),
    tournamentSize(tournamentSize),
    maxGenerations(maxGenerations),
    generator(std::random_device{}())
{};

std::vector<KnapsackGA::Solution>GeneticAlgorithm::generateInitialPopulation(){
    std::vector<KnapsackGA::Solution> population;
    std::uniform_int_distribution<> boolDist(0, 1);

    for (int i = 0; i < this->populationSize; i++)
    {
        KnapsackGA::Solution chromosome(this->chromosomeLength);
        for (int j = 0; j < this->chromosomeLength; j++)
        {
            chromosome[j] = boolDist(this->generator);
        }
        population.push_back(chromosome);
    }
    
    return population;
}

int GeneticAlgorithm::calculateFitness(KnapsackGA::Solution& chromosome){
    auto [totalWeight, totalValue] = this->knapsack.evaluate(chromosome);

    if(totalWeight > knapsack.getCapacity()){
        return 0;
    }

    return totalValue;
}

KnapsackGA::Solution GeneticAlgorithm::tournamentSelection(std::vector<KnapsackGA::Solution>& population){
    std::uniform_int_distribution<> indexDist(0, this->populationSize - 1);
    KnapsackGA::Solution bestChromosome;
    int bestFitness = -1;

    for (int i = 0; i < this->tournamentSize; i++)
    {
        int index = indexDist(this->generator);
        KnapsackGA::Solution chromosome = population[index];
        int fitness = this->calculateFitness(chromosome);
        if (fitness <= bestFitness)
            continue;
        bestFitness = fitness;
        bestChromosome = chromosome;
    }
    return bestChromosome;
}

std::pair<KnapsackGA::Solution, KnapsackGA::Solution> GeneticAlgorithm::crossover(KnapsackGA::Solution& parent1, KnapsackGA::Solution& parent2){
    std::uniform_int_distribution<> pointDist(1, this->chromosomeLength);

    int point = pointDist(this->generator);
    KnapsackGA::Solution child1 = parent1;
    KnapsackGA::Solution child2 = parent2;

    for (int i = point; i < this->chromosomeLength; i++)
    {
        // std::swap(child1[i], child2[i]);
        std::vector<bool>::swap(child1[i], child2[i]);
    }
    return {child1, child2};
}

void GeneticAlgorithm::mutate(KnapsackGA::Solution& chromosome){
    std::uniform_real_distribution<> probabilityDist(0.0, 1.0);
    for (int i = 0; i < this->chromosomeLength; i++)
    {
        if(probabilityDist(this->generator) >= this->mutationRate)
            continue;
        chromosome[i] = !chromosome[i];
    }
}

std::pair<KnapsackGA::Solution, KnapsackGA::Solution> GeneticAlgorithm::solve(bool verbose, bool log, bool isParallel){
    std::vector<KnapsackGA::Solution> population = this->generateInitialPopulation();
    KnapsackGA::Solution bestSolution;
    KnapsackGA::Solution bestLastGenSolution;
    int bestFitness = -1;
    std::vector<int> convergence;

    for (int generation = 0; generation < this->maxGenerations; generation++)
    {
        std::vector<int> fitnesses(this->populationSize);
        int bestGenFitness = -1;
        KnapsackGA::Solution bestGenSolution;

        #pragma omp parallel for if(isParallel)
        for (int i = 0; i < this->populationSize; ++i)
        {   
            int fitness = this->calculateFitness(population[i]);
            fitnesses[i] = fitness;
        }

        int bestGenIndex = -1;
        for (int i = 0; i < population.size(); ++i) {
            if (fitnesses[i] > bestGenFitness) {
                bestGenFitness = fitnesses[i];
                bestGenIndex = i;
            }
        }

        bestGenSolution = population[bestGenIndex];

        if(bestGenFitness > bestFitness){
            bestFitness = bestGenFitness;
            bestSolution = bestGenSolution;
        }

        std::vector<KnapsackGA::Solution> newPopulation(this->populationSize);
        newPopulation[0] = bestGenSolution;

        #pragma omp parallel for if(isParallel)
        for(int i = 1; i < this->populationSize; i += 2){
            std::mt19937 localGen(std::random_device{}() + i);

            KnapsackGA::Solution parent1 = this->tournamentSelection(population);
            KnapsackGA::Solution parent2 = this->tournamentSelection(population);

            auto [child1, child2] = this->crossover(parent1, parent2);

            mutate(child1);
            mutate(child2);

            newPopulation[i] = child1;
            if(i + 1 < this->populationSize){
                newPopulation[i+1] = child2;
            }
        }

        population = newPopulation;
        bestLastGenSolution = bestGenSolution;
        convergence.push_back(bestGenFitness);
    }

    return {bestSolution, bestLastGenSolution};
}
