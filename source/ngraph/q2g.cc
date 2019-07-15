#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include "ngraph.hpp"
#include "ngraph_algorithms.cc"
#include "token.hpp"
/* 
 This converts a clique graph into a regular [undirected] graph

  The form of the clique graph is one clique per line

  node_a node_b node_c
  node_e node_f
  node_g
  node_h node_k



*/

using namespace std;
using namespace NGraph;

int main(int argc, char *argv[])
{

  Graph B;
  bool undirected = false;
  string input_line;

  if (argv[1])
  {
    std::string arg1(argv[1]);
    undirected = (arg1 == "-u");
  }



  while (getline(cin, input_line))
  {

      // for each line, read all nodes in clique
      vector<Graph::vertex> nodes;
      Tokenizer T(input_line);
      string next_node;
      while ( ! (next_node = T.next()).empty() )
      {
        stringstream s(next_node);
        Graph::vertex i;
        s >> i;
        nodes.push_back(i);
      }

      if (undirected)
        add_undirected_clique(B, nodes.begin(), nodes.end());
      else
        add_clique(B, nodes.begin(), nodes.end());

  }

  std::cout << B;

  return 0;
}
