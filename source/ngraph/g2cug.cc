// Converts a regular '*.g' file into a compact undirected graph.
// (Stores only (i,j) where i < j and 


#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <cmath>
#include "ngraph.hpp"

//
//  Usage g2cug < graph.g  > graph_undirected.g
//

using namespace std;
using namespace NGraph;

int main(int argc, char *argv[])
{


  Graph G;

  Graph::vertex v1, v2;
  bool vertex_only = false;

  while (G.read_line(std::cin, v1, v2, vertex_only))
  {
      if (vertex_only)
      {
        G.insert_vertex(v1);
      }
      else
        G.insert_edge(v1,v2);
  }

  std::cout << G ;

  return 0;
}


