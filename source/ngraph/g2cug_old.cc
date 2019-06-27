// Converts a regular '*.g' file into a compact undirected graph.
// (Stores only (i,j) where i < j and 


#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <cmath>
#include "ngraph.hpp"

//
//  Usage a.out < graph.g    
//

using namespace std;
using namespace NGraph;

int main(int argc, char *argv[])
{


  Graph A;

  while (!std::cin.eof())
  {
     Graph::vertex v1, v2;

     std::cin >> v1 >> v2;
     if (v1 < v2)
      A.insert_edge(v1, v2);
     else if (v1 > v2)
      A.insert_edge(v2, v1);

  }


 std::cout << A << "\n";

}

