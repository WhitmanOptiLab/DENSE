#include <iostream>
#include <string>
#include <sstream>
#include "ngraph.hpp"
#include "ngraph_algorithms.cc"
#include "ngraph_cluster_coeff.cc"

using namespace std;

int main(int argc, char *argv[])
{

  int reps =1;
  Graph A;

  bool undirected = false;
  if (argv[1])
  {
    std::string arg1(argv[1]);
    undirected = (arg1 == "-u");
    istringstream buffer(arg1);
    buffer >> reps;
  }

  while (!std::cin.eof())
  {
     int v1, v2;

     std::cin >> v1 >> v2;
     // for an undirected graph
     A.insert_edge(v1, v2);
     A.insert_edge(v2, v1);
  }

#if 0
    std::cout << "Avg Degree corrleation: "<<avg_degree_correlation(A) << "\n";
    std::cout << "Max Degree corrleation: "<<max_degree_correlation(A) << "\n";
    std::cout << "Max sqrt  corrleation: "<<max_sqrt_degree_correlation(A) << "\n";
#endif

   unsigned int n_edges = (undirected) ? A.num_edges()/2 : A.num_edges();

    std::cout << A.num_vertices() << " " << n_edges <<  " " 
        << cluster_coeff(A) << " " << avg_degree_correlation(A) << "\n";

  return 0;
}

