#include "ngraph.hpp"
#include "ngraph_conductance.hpp"

using namespace NGraph;
using namespace std;

int  main()
{

    Graph G;
    cin >> G;


    Graph::vertex_set V;

    // find the conductance of the first N/2 nodes
    // int half_nodes = G.num_vertices() / 2;

    unsigned int N = 5;

    Graph::const_iterator p = G.begin();

    for (int i=0; i< N; i++)
    {
        V.insert( Graph::node(p) );
        p++;
    }
    cout <<  "allignment: " << alignment(G, V) << "\n";
    cout <<  "conductance: " << conductance(G, V) << "\n";

    return 0;

}
