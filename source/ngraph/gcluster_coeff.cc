#include <iostream>
#include "ngraph.hpp"
// #include "ngraph_cluster_coeff.hpp"

using namespace std;
using namespace NGraph;


//template <typename T>
double cluster_coeff_directed(const Graph &G, Graph::const_iterator t)
{

	const  Graph::vertex_set &neighbors = 
            Graph::in_neighbors(t) +  Graph::out_neighbors(t);

	int n = neighbors.size();

	if (n < 2) return 0.0;

	int num_edges = 0;
	//for each vertex in neighbors of a
	//
	for ( Graph::vertex_set::const_iterator p=neighbors.begin() ; 
					p != neighbors.end(); p++ )
	{
		num_edges += intersection_size(G.out_neighbors(*p),neighbors); 
	}
  return  static_cast<double>(num_edges) / (n*(n-1));
}



int main()
{

  Graph G;

  cin >> G;

  for (Graph::const_iterator p=G.begin(); p!=G.end(); p++)
  {
      cout << Graph::node(p) << " " << 
            cluster_coeff_directed(G,p) << "\n";
  }


  return 0;
}

