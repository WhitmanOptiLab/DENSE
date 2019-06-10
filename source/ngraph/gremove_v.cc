//  gremove_v
//
//  remove a lit of vertices from a graph
//
//  Usage:  cat nodes_to_remove.v | gremove_v  graph.g >  graph_without_nodes.g
//

#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include "ngraph.hpp"
#include "set_ops.hpp"

#define foreach(p,C)   for (p=C.begin(); p!=C.end(); (p)++)

using namespace std;
using namespace NGraph;

int main(int argc, char *argv[])
{

  if (argc <= 1)
  {
    std::cerr << "Usage :   cat nodes.v  | gremove_v foo.g > foo_small.g \n";
    exit(1);
  }

  istream &vertex_file = cin;

  set<Graph::vertex> V;

   // read in vertices to remove
   while(! vertex_file.eof())
   {
      Graph::vertex v;
      vertex_file >> v;
      //cerr << v << "\n";
      V.insert(v);
   }


  // Now remove them from in-coming graph

  const char *graph_filename = argv[1];
  ifstream graph_file;
  graph_file.open(graph_filename);
  Graph G;
  graph_file >> G;
  graph_file.close();

  G.remove_vertex_set(V);
  cout << G;

  return 0;
}
