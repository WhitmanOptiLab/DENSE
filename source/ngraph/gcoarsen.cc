/**

Coarsen a graph by collapsing a group of nodes into one supernode.
Usage is

  cat nodes.txt | gcoarsen foo.g > foo_coarsen.g

where the input is a list of clusters (one per line), where each 
cluster is a list of vertices


*/


#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <map>
#include <set>
#include "ngraph.hpp"

using namespace std;
using namespace NGraph;

typedef map<unsigned int, string> node_array_type;

int main(int argc, char *argv[])
{

  Graph G;

  ifstream graph_file;
  graph_file.open(argv[1]);
  if (!graph_file) 
  {
    exit(1);
  }
 
  graph_file >> G;

  // now read cluster list, one cluster per line ...
  // and aboorb the first vertex in the cluster with all the other ones
  
  string line;
  while ( getline(cin, line))
  {
      Graph::vertex v;
      Graph::vertex v1;
      stringstream s(line);
      if ( !(s >> v1) )
          break;
      while (s >> v)
      {
        G.smart_absorb(v1, v);
      }
  }

  cout << G;

  return 0;
}

