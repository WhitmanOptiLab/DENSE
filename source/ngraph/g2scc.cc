#include <iostream>
#include "ngraph_scc.cc"

int main()
{

    Graph G;

    std::cin >> G;

//    std::cout << "Graph: \n" << G << "\n";

    SCC SCC_G(G);

#if 0
    for (Graph::const_iterator p=G.begin(); p!=G.end(); p++)
    {
      Graph::vertex v = Graph::node(p);
      std::cout << "SCC[" << v << "] = " << SCC_G.component(v) << "\n"; 
    }
#endif

    return 0;
}

