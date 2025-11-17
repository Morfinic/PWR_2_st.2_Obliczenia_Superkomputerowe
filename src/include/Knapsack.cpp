#include "Knapsack.h"


Knapsack::Knapsack(int capacity): capacity(capacity) {};

void Knapsack::addItem(Item& item){
    this->items.push_back(item);
}

int Knapsack::getCapacity(){
    return this->capacity;
}

int Knapsack::getItemCount(){
    return this->items.size();
}

KnapsackGA::Items Knapsack::getItems(){
    return this->items;
}

KnapsackGA::KnapsackEvaluation Knapsack::evaluate(KnapsackGA::Solution& solution){
    int totalWeight = 0;
    int totalValue = 0;

    for (size_t i = 0; i < solution.size(); i++)
    {
        if(!solution[i])
            continue;
        totalWeight += this -> items[i].weight;
        totalValue += this -> items[i].value;
    }
    
    return {totalWeight, totalValue};
}

Knapsack Knapsack::generateRandom(int numItems, int maxWeight, int maxValue, int capacity, bool fixed){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> weightDist(1, maxWeight);
    std::uniform_int_distribution<> valueDist(1, maxValue);
    int weight = 0;
    int value = 0;

    Knapsack knapsack(capacity);

    KnapsackGA::Items items;
    for (int i = 0; i < numItems; i++)
    {
        if(fixed){
            weight = weight + 10;
            value = value + 10;
        }
        else{
            weight = weightDist(gen);
            value = valueDist(gen);
        }
        Item item(i, weight, value);
        knapsack.addItem(item);
    }
    
    return knapsack;
}

void Knapsack::printSolution(KnapsackGA::Solution& solution){
    auto [totalWeight, totalValue] = this->evaluate(solution);

    std::cout << "Selected items: " << std::endl;
    int count = 0;

    for (size_t i = 0; i < solution.size(); i++)
    {
        if(!solution[i])
            continue;
        std:: cout << this -> items[i] << std::endl;
        count++;
    }
    std::cout << std::endl;
    std::cout << "Total items: " << count << std::endl;;
    std::cout << "Total weight: " << totalWeight << " / " << this -> capacity << std::endl;
    std::cout << "Total value: " << totalValue << std::endl;
    std::cout << "----------------" << std::endl;
}