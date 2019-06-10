/*
    Given a list of vertices, print their in and out-neighbor 


    Usage:  cat math.v | gneighbors_v  math.g

            echo '14 67 89' | gneighbors_v foo.g

The output is the node number, with [in] and [out], eg.

     14  [7 9 12 45 78] [6 34 99]
     17  [ 2  4]  [ 7]
     ...
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
 
    typedef set<Graph::vertex>  vertex_set;

    if (argc < 2)
    {
        cerr << "Usage: cat foo.v | " << argv[0] << " foo.g \n";
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

   Graph G;
   graph_file >> G;


    for (vertex_set::const_iterator p = V.begin(); p!= V.end(); p++)
    {
        Graph::vertex v = *p;
        cout << v  << " [ " ;
        const Graph::vertex_set &in = G.in_neighbors(v);
        for (vertex_set::const_iterator vi=in.begin(); vi != in.end(); vi++)
        {
          cout << Graph::node(vi) <<" " ;
        }
        cout << "]  [" ;

        const Graph::vertex_set &out = G.out_neighbors(v);
        for (vertex_set::const_iterator vo=out.begin(); vo!=out.end(); vo++)
        {
          cout << Graph::node(vo) <<" " ;
        }
        cout << "]\n";
    }

    
    return 0;
}
