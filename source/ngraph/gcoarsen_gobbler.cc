#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <cmath>
#include "set_ops.hpp"
#include "ngraph.hpp"


// this is an old version of gprune (actually, gobbler)
//
//  Usage  cat foo.g |  gcoarsen [-u] > reduced_foo.g
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

  size_t max_string_len = 0;

  while (!std::cin.eof())
  {
     Graph::vertex v1, v2;

     std::cin >> v1 >> v2;

     A.insert_edge(v1, v2);
     if (undirected)
        A.insert_edge(v2, v1);
  }


   

  // build up all of the vertices of A.  (We can't destroy these in
  // place as we are traversing the graph.)
  //
  Graph::vertex_set V;
  for ( Graph::iterator p = A.begin(); p!=A.end(); p++)
  {
      V  += Graph::node(p);
  }

  // now we make multiple passes until we can no longer coarsen the
  // graph.

  bool no_changes = False;

  set<Graph::vertex> deleted_nodes;

  do 
  {

    // loop over the remaining vertices

    for (Graph::vertex_set::iterator p = V.begin(); p !=V.end(); p++)
    {
       Graph::vertex here = *p;
       if ( A.in_neighbors(here).size() == 1 )
       {
          Graph::vertex root =  *(A.in_neighbors(here).begin()); 
          deleted_nodes += root;
          A.collapse(root, here);
          cerr << "Collaped " << here << " into " << root << ".\n";
       }
    }

    V -= deleted_nodes;
  }
  while ( deleted_nodes.size() > 0)


  cout << A;

}

