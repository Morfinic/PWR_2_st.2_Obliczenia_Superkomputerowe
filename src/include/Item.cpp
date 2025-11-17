#include "Item.h"

Item::Item(int id, int weight, int value): id(id), weight(weight), value(value) {};

std::ostream& operator<<(std::ostream& os, const Item& item) {
    os << "Item " << item.id << ": Weight = " << item.weight << ", Value = " << item.value;
    return os;
}
