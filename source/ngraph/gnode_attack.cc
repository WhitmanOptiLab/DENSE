#include <fstream>
#include <iostream>
#include <string>
#include <set>
#include "set_ops.hpp"
#include "ngraph.hpp"
#include "ngraph_components.hpp"
#include "equivalence.hpp"

/*
    Given a list of nodes, this program lists the number of connected
    components remaining in the graph.

    Usage:  cat nodes_to_remove.v | gnode_attack

    output:  one line per node_removed

*/

using namespace std;
using namespace NGraph;

int main(int argc, char *argv[])
{
   Graph G;

   if (argc < 2)
   {
      cerr << "Usage: "<< argv[0] << " graph.g < graph.v \n";
      exit(1);
   }

   const char *graph_filename = argv[1];
   ifstream graph_file;
   graph_file.open( graph_filename);
    if (!graph_filename)
    {
        cerr << "Error: [" << graph_filename << "] could not be opened.\n";
        exit(1);
    }
    graph_file >> G;


   // Now find number of connected components

   typedef equivalence<Graph::vertex> equiv;

   Graph::vertex node_to_remove;
   while (cin>>node_to_remove)
   {
      G.remove_vertex(node_to_remove);  
      equiv E = components(G);

      // compute the min, max, and average component sizes

      unsigned int Min=0, Max=0;
      double Avg=0.0;

      equiv::const_iterator p=E.begin();
      if (p != E.end())
      {
          unsigned int class_size =  equiv::collection(p).size();
          Min = Max = class_size;
          Avg += class_size;
          p++;
      }
      for (; p!=E.end(); p++)
      {
          //const equiv::element_set &V = equiv::collection(p);
          unsigned int class_size =  equiv::collection(p).size();
          if (class_size > Max)  Max = class_size;
          if (class_size < Min) Min = class_size;
          Avg += class_size;
      }
      Avg /= E.size();
      cout << node_to_remove << " " << E.size() << " " << Min << 
              " " << Avg << " " << Max << "\n";;
   }


   return 0;
}
