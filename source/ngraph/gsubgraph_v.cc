/*
    Given a grah file (.g) from stdin, create a sugraph of only those nodes 
    listed in a vertex file (.v)

    Usage:  cat math.v | subgraph_v math.g

*/


#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include "ngraph.hpp"
#include "set_ops.hpp"

using namespace std;
using namespace NGraph;


int main(int argc, char *argv[])
{
  
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " graph.g < graph.v \n";
        exit (1);
    }

    const char *graph_filename = argv[1];
    
    istream &vertex_file = cin;
    ifstream graph_file;
    
    //cerr << "Opening ["<< graph_filename << "]" << endl;

    graph_file.open( graph_filename);
    if (!graph_file)
    {
        cerr << "Error: [" << graph_filename << "] could not be opened.\n";
        exit(1);
    }

    set<Graph::vertex> V;

    // read in vertices and build set
    while (!vertex_file.eof())
    {
        Graph::vertex node;

        vertex_file >> node ;
        V.insert(node);
    }

   Graph G;
   graph_file >> G;

   Graph G_V = G.subgraph(V);

    cout << G_V;

    return 0;
}
