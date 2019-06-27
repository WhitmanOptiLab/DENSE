#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include "ngraph.hpp"
#include "ngraph_conductance.hpp"

/*
    compute the alignment of clusters (one per line) in a given graph.

    If cluster.v is a line-separated list of clusters, then

    cat cluster.v | g2alignment foo.g > foo.alignment

    prints out each cluster's alginment (similar to conductance) in
    the larger file foo.g.

*/

using namespace std;
using namespace NGraph;

int main(int argc, char *argv[])
{
    ifstream graph_file;
    graph_file.open(argv[1]);
    if (!graph_file)
    {
        exit(1);
    }

    Graph G;
    graph_file >> G;
    

    std::string line;
    while (getline(cin, line))
    {
          stringstream s(line);

          Graph::vertex word;
          Graph::vertex_set V;
          
          while (s >> word)
          {
              V.insert(word);
          }
          cout << cut_size(G,V) << " " << conductance(G,V) << " " << alignment(G, V);
          cerr << ": [" << line << "]";
          cout << "\n";
     }


  return 0;
}

