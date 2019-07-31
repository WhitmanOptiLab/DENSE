/**

    Create the weakly-connected components of an input graph, given a specific
    order of nodes (one, or several per line) and report number of components
    at each step.

    Usage:

      cat node.list | g2weak foo.g  


    Input:  graph (foo.g) and a node list with one or more vertices 
    per line, e.g.


    223  29  2  1 13
    334
    87 43 7
    ...

    The vertices represent some order (e.g. degree, page-rank, etc.) to build
    the subgraph.

    Output:  the number of weakly-connected components (each number 
    coresponds to the original node list, e.g.

    3  
    4
    4
    ...

*/

#include <iostream>
#include <fstream>
#include <list>
#include <set>
#include <string>
#include <algorithm>
#include <numeric>
#include "ngraph.hpp"
#include "ngraph_weak.hpp"



using namespace std;
using namespace NGraph;

int main(int argc, char *argv[])
{

   if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " graph.g < node.list \n";
        exit (1);
    }

    const char *graph_filename = argv[1];

    ifstream graph_file;
    graph_file.open( graph_filename);
    if (!graph_filename)
    {
        cerr << "Error: [" << graph_filename << "] could not be opened.\n";
        exit(1);
    }



    Graph G;
    graph_file >> G;
    graph_file.close();

    equivalence<Graph::vertex> Q;
    set<Graph::vertex> V;
    string input_line;

    while (getline(cin, input_line))
    {
        Graph::vertex v;
        list<Graph::vertex> L;
        stringstream I(input_line);
        
        while (I >> v)
        {
          L.push_back(v);
          //cout << v << " ";
          V.insert(v);
        }
        //cout << "\n" ;

        // now find weak component set of this subgraph

        weak_components_increment(G, Q, V, L.begin(), L.end());


        vector<unsigned int> V = Q.class_sizes();

        unsigned int Max_size = *max_element(V.begin(), V.end());

        cout << Q.num_classes() << " " << Max_size << " " << Q.num_elements();
        //cout << << ": \n" << Q ;
        cout << "\n";
    }

    return 0;
      
}
