/**

    Create the weakly-connected components of an input graph, given a max
    degree and report all clusters of minimum-size.

    Usage:

      cat foo.g | g2Pmetric_level [min-cluster_size] [max-degree]   


    Output:  the clusters, seperaeted by blank links, with one one node per 
    line.



*/

#include <iostream>
#include <fstream>
#include <list>
#include <set>
#include <string>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include "ngraph.hpp"
#include "ngraph_weak.hpp"



using namespace std;
using namespace NGraph;

int main(int argc, char *argv[])
{

    if (argc <= 2)
      exit(1);

    bool line_output = false;
    int arg_num = 1;
    if (string(argv[arg_num]) == "-l")
    {
        line_output = true;
        arg_num++;
    }
    const unsigned int min_cluster_size = atoi(argv[arg_num++]);
    const unsigned int degree_level = atoi(argv[arg_num++]);


    Graph G;
    std::cin >> G;

    typedef Graph::vertex Vertex;

    // create a frequency distribution of nodes based on degree
    typedef map< unsigned int, set<Graph::vertex> > DegreeList;
    DegreeList D;


    for ( Graph::const_iterator p = G.begin(); p!=G.end(); p++)
    {
      D[Graph::in_degree(p) + Graph::out_degree(p)].insert(Graph::node(p));
    }


    equivalence<Graph::vertex> Q;
    set<Graph::vertex> V;
    string input_line;

    for (DegreeList::const_iterator p=D.begin(); (p->first <= degree_level); 
                p++)
    {
     
        const set<Graph::vertex> &L = p->second;
        V.insert(L.begin(), L.end());

        // now find weak component set of this subgraph

        weak_components_increment(G, Q, V, L.begin(), L.end());

    }

    typedef equivalence<Graph::vertex>::const_iterator cluster_iterator;

    for (cluster_iterator p=Q.begin(); p!=Q.end(); p++)
    {
        const set<Graph::vertex> &c = p->second;
        if (c.size() >= min_cluster_size)
        {
          if (line_output) // print one cluster per line
          {
            for ( set<Vertex>::const_iterator v = c.begin(); v != c.end(); v++)
            {
              cout << *v << " ";
            }
          }
          else  // print cluster, one node per line
          {
            for ( set<Vertex>::const_iterator v = c.begin(); v != c.end(); v++)
            {
              cout << *v << "\n";
            }
          }
          cout << "\n";
        }
    }


    return 0;
      
}
