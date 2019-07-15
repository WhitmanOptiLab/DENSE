// Converts a regular '*.g' file into a compact undirected graph.
// (Stores only (i,j) where i < j and does not store (i,i)


#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <cmath>
#include "nigraph.cc"

//
//  Usage a.out < graph.g    
//

using namespace std;

template <class T>
T max(T x1, T x2, T x3)
{
  T max1 =  (x1 > x2) ? x1 : x2;
  return (max1 > x3) ?  max1 : x3;
  
}


int main(int argc, char *argv[])
{


  Graph A;
  int max_node = 0;

  while (!std::cin.eof())
  {
     int v1, v2;

     std::cin >> v1 >> v2;
     if (v1 >= v2)
      A.insert_edge(v1, v2);
     else 
      A.insert_edge(v2, v1);

    max_node = max( max_node, v1, v2);
  }

 std::cout << "%%MatrixMarket matrix coordinate pattern symmetric\n";
 std::cout << max_node << "  " << max_node << " " << A.num_edges() << "\n";
 std::cout << A << "\n";

}

