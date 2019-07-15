#include <iostream>
#include <string>
#include <set>
#include "set_ops.hpp"
#include "ngraph.hpp"
#include "ngraph_components.hpp"
#include "equivalence.hpp"

/*
    Extract largest weakly connected componnent out of grahp.

    Usage:  cat graph.g | g2giant_component > graph.v

*/

using namespace std;
using namespace NGraph;

int main(int argc, char *argv[])
{
   Graph G;
   bool verbose = false;
   bool line_verbose = false;
   bool print_largest = false;

  typedef unsigned int uInt;

   if (argc > 1)
   {
      std::string arg1(argv[1]);
      verbose = (arg1 == "-v");
      line_verbose = (arg1 == "-lv" || arg1 == "-lv");
      print_largest = (arg1 == "-b");
   }

   cin >> G;

   // Now find number of connected components

   typedef equivalence<Graph::vertex> equiv;
   equiv E = components(G);



   // Find the largest componnent (equivalence class)

   uInt max_index  = 0;
   uInt max = 0;
   for (uInt i=0; i<E.num_classes(); i++)
   {
      if (E.class_size(i) > max)
      {
          max = E.class_size(i);
          max_index = i;
      }
   }

  // Now E has the equivalence class of graph nodes that correspond to
  // connected graphs. The size of each partition denotes how many 
  // vertices it has. Then, we use the subgraph_size() method to find
  // out how many edges are in each connecgted component.
  //

  const set<Graph::vertex> &e = E[max_index];
  typedef set<Graph::vertex>::const_iterator iter;
  for (iter p=e.begin(); p!=e.end(); p++)
  {
      cout << *p << " ";
  }
 cout << "\n";

  return 0;
}
