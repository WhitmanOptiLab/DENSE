/**

    Using nodes of degree less than or equal to d, return the largest 
    (weakly) connected component.

    Usage:

      cat foo.g | gbigcomponent  [max-degree]   


    Output:  nodes of largest components, one per line.



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

    if (argc <= 1)
      exit(1);

    const unsigned int degree_level = atoi(argv[1]);


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

    unsigned int max_cluster_size = 0;
    cluster_iterator pm = Q.begin();
    for (cluster_iterator p=Q.begin(); p!=Q.end(); p++)
    {
        const set<Graph::vertex> &c = p->second;
        if (c.size() > max_cluster_size)
        {
            max_cluster_size = c.size();
            pm = p;
        }
    }
    
    if (max_cluster_size > 0)
    {

        const set<Graph::vertex> &c = pm->second;
        {
            // print cluster, one node per line
            for ( set<Vertex>::const_iterator v = c.begin(); v != c.end(); v++)
            {
              cout << *v << "\n";
            }
            cout << "\n";
        }
    }


    return 0;
      
}
