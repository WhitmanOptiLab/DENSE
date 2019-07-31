// Reverses directions of edges in a directed graph.


#include <iostream>
#include <string>

//
//  Usage a.out < graph.g    
//

//using namespace std;

int main(int argc, char *argv[])
{


  std::string node1, node2;

#if 0
  while (!std::cin.eof())
  {

     std::cin >> node1 >> node2;
     std::cout << node2 <<  " " << node1 << "\n";

  }

#endif
  
  
  while (std::cin >> node1 >> node2)
  {

     std::cout << node2 <<  " " << node1 << "\n";

  }

}

