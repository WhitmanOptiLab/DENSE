#include "ngraph.hpp"
#include "ngraph_conductance.hpp"

using namespace NGraph;
using namespace std;

int  main()
{

    Graph G;
    cin >> G;

    // find the conductance of the first N/2 nodes

    Graph::vertex_set V;
    int half_nodes = G.num_vertices() / 2;
    Graph::const_iterator p = G.begin();

    for (int i=0; i< half_nodes; i++)
    {
        V.insert( Graph::node(p) );
        p++;
    }
    cout <<  conductance(G, V) << "\n";

    return 0;

}
