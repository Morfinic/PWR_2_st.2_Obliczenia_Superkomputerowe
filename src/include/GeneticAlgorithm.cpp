#include "GeneticAlgorithm.h"

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

std::pair<KnapsackGA::Solution, KnapsackGA::Solution> GeneticAlgorithm::solve(bool verbose, bool log){
    std::vector<KnapsackGA::Solution> population = this->generateInitialPopulation();
    KnapsackGA::Solution bestSolution;
    KnapsackGA::Solution bestLastGenSolution;
    int bestFitness = -1;
    std::vector<int> convergence;

    for (int generation = 0; generation < this->maxGenerations; generation++)
    {
        std::vector<int> fitnesses;
        int bestGenFitness = -1;
        KnapsackGA::Solution bestGenSolution;
        
        for (auto &chromosome : population)
        {   
            int fitness = this->calculateFitness(chromosome);
            fitnesses.push_back(fitness);

            if(fitness <= bestGenFitness)
                continue;
            bestGenFitness = fitness;
            bestGenSolution = chromosome;
        }

        if(bestGenFitness > bestFitness){
            bestFitness = bestGenFitness;
            bestSolution = bestGenSolution;
            if (verbose){
                std::cout << "Generation " << generation << " New best solution " << bestFitness << std::endl;
                std::cout << "Knapsack " << std::endl; 
                knapsack.printSolution(bestSolution);
            }
        }

        std::vector<KnapsackGA::Solution> newPopulation;
        newPopulation.push_back(bestGenSolution);
        while(newPopulation.size() < this->populationSize){
            KnapsackGA::Solution parent1 = this->tournamentSelection(population);
            KnapsackGA::Solution parent2 = this->tournamentSelection(population);

            auto [child1, child2] = this->crossover(parent1, parent2);

            mutate(child1);
            mutate(child2);

            newPopulation.push_back(child1);
            if(newPopulation.size() < this->populationSize){
                newPopulation.push_back(child2);
            }
        }

        population = newPopulation;
        bestLastGenSolution = bestGenSolution;
        convergence.push_back(bestGenFitness);
    }
    if(log){
        std::ofstream file("results/convergence.csv");
        file << "generation,bestFitness\n";
        for (int i=0; i < convergence.size(); i++)
        {
            file << i+1 << "," << convergence.at(i) << "\n";
        }
    }
    return {bestSolution, bestLastGenSolution};
}
