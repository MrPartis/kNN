#include "kNN.hpp"

void tc1(){
Dataset dataset;
dataset.loadFromCSV("mnist.csv");
List<int> *row = dataset.getData()->get(0);

row->push_back(1);
row->push_back(2);
row->push_back(1);

cout << row->length() << endl;
row->print();
}

int main() {
    tc1();
    return 0;
}
