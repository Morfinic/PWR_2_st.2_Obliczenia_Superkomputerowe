#pragma once
#include <iostream>

class Item
{
    public:
        int id;
        int weight;
        int value;

        Item(int id, int weight, int value);
        friend std::ostream& operator<<(std::ostream& os, const Item& item);
};
