#include <iostream>
#include <vector>
#include "equivalence.hpp"

using namespace std;

int main()
{

    equivalence<char> E;


    E.insert('a');
    E.insert('b');
    E.insert('c', 'd');
    E.insert('e', 'a');
    E.insert('f', 'g');
    
    cout << "equivalence set has " << E.num_classes() << 
          " classes and " << E.num_elements() << " elements.\n";
    cout << E << "\n";

    E.insert('h', 'd');
    E.insert('c', 'a');

    cout << "equivalence set has " << E.num_classes() << 
          " classes and " << E.num_elements() << " elements.\n";
    cout << E << "\n";

    E.insert('a','f');
    E.insert('b','g');
    cout << "equivalence set has " << E.num_classes() << 
          " classes and " << E.num_elements() << " elements.\n";
    cout << E << "\n";


#if 0
    vector<int> v;

    v.push_back(20);
    v.push_back(30);
    v.push_back(40);
    v.push_back(50);

    E.insert(v.begin(), v.end());
    cout << "equivalence set has " << E.num_classes() << 
          " classes and " << E.num_elements() << " elements.\n";
    cout << E << "\n";
#endif


    return 0;
}



