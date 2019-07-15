/*
    Given a grah file (.g) from stdin, create a sugraph of only those nodes 
    listed in a vertex file (.v)

    Usage:  cat math.v | subgraph math.g

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
    if (!graph_filename)
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

    while (!graph_file.eof())
    {
       Graph::vertex  to, from;
       graph_file >> to >> from;
       bool includes_to = includes_elm(V, to);
       bool includes_from = includes_elm(V, from);

       if (includes_to && includes_from)
          cout << to << " " << from << "\n";
       else if (includes_to)
       {
         cout << to << "\n";
       }
       else if (includes_from)
       {
         cout << from << "\n";
       }

    }
    graph_file.close();



    return 0;
}
