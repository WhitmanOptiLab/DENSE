#include <iostream>
#include "set_ops.hpp"

int main()
{
    std::set<int> A, B;

    A.insert(1); A.insert(2); A.insert(4); A.insert(7);
    B.insert(2); B.insert(3); B.insert(5); B.insert(7);

    A += B;

    for (std::set<int>::const_iterator p=A.begin(); p!=A.end(); p++)
    {
      std::cout << *p << std::endl;
    }
    std::cout << std::endl;

    A -= B;
    for (std::set<int>::const_iterator p=A.begin(); p!=A.end(); p++)
    {
      std::cout << *p << std::endl;
    }
    std::cout << std::endl;



   return 0;
}
