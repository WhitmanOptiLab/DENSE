#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <cmath>
#include "ngraph.hpp"
#include "ngraph_cluster_coeff.cc"

//
//  Usage a.out [-u] < graph.dat     (-u for undirected)
//

using namespace std;
using namespace NGraph;

int main(int argc, char *argv[])
{


  Graph A;

  bool undirected = false;
  if (argv[1])
  {
    std::string arg1(argv[1]);
    undirected = (arg1 == "-u");
  }

  while (!std::cin.eof())
  {
     int v1, v2;

     std::cin >> v1 >> v2;
     A.insert_edge(v1, v2);
     if (undirected)
        A.insert_edge(v2, v1);
  }


   unsigned int print_width =  static_cast<unsigned int>(
                          ceil(log10(A.num_nodes()) + 2));
   const unsigned int coeff_width = 4;

  for ( Graph::const_iterator p = A.begin(); p!=A.end(); p++)
  {
      std::cout <<  left 
                << setw(print_width) <<  Graph::node(p);

      if (undirected)
      {
         std::cout << setw(print_width) <<  Graph::in_neighbors(p).size(); 
      }
      else
      {
         std::cout << setw(print_width) <<  Graph::in_neighbors(p).size() 
                << setw(print_width) <<  Graph::out_neighbors(p).size() ;
      }
      std::cout << fixed 
                // << setw(coeff_width+2) 
                << setprecision(coeff_width)
                <<  cluster_coeff_directed(A,  Graph::node(p)) 
                << "\n";
      std::cout.flush();
  }

}

