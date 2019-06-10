#include <iostream>
#include <string>
#include <set>
#include "set_ops.hpp"
#include "ngraph.hpp"
#include "ngraph_components.hpp"
#include "equivalence.hpp"

/*
    Lists sizes (V,E) of each connected component in a graph.

    Usage:  cat graph.g | gcomponents  [-v | -lv | -b ]

   -v  verbose (print one component per line)
   -lv verbose (print one vertex per line, with components separated by newline
   -b  print vertices of largest component, one per line

*/

using namespace std;
using namespace NGraph;

int main(int argc, char *argv[])
{
   Graph G;
   bool verbose = false;
   bool line_verbose = false;
   bool print_largest = false;

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

  // Now E has the equivalence class of graph nodes that correspond to
  // connected graphs. The size of each partition denotes how many 
  // vertices it has. Then, we use the subgraph_size() method to find
  // out how many edges are in each connecgted component.
  //

  if (verbose)
  {
    for (equiv::const_iterator p = E.begin(); p!=E.end(); p++)
    {
       const equiv::element_set &V = equiv::collection(p); 
       for(equiv::element_set::const_iterator pV=V.begin(); pV!=V.end();pV++)
            cout << *pV << " ";
       cout << "\n";
    }
  }
  else
  {
    if (line_verbose)
    {
       for (equiv::const_iterator p = E.begin(); p!=E.end(); p++)
       {
         const equiv::element_set &V = equiv::collection(p); 
         for(equiv::element_set::const_iterator pV=V.begin(); pV!=V.end();pV++)
            cout << *pV << "\n";
         cout << "\n";
        }
    }
    else    // just print out num_nodes and edges for each conponenet
    {
      for (equiv::const_iterator p = E.begin(); p!=E.end(); p++)
      {
        const equiv::element_set &V = equiv::collection(p); 
        //unsigned int i = equiv::index(p);
        //cout << i << " " << V.size() << " " << G.subgraph_size(V)<< "\n";
        cout << " " << V.size();
        //cout <<" " << G.subgraph_size(V);
        cout <<  "\n";
      }
     }
  }

   return 0;
}
