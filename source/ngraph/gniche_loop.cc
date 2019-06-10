#include <iostream>
#include <vector>
#include <algorithm>
#include "ngraph.hpp"
#include "equivalence.hpp"

typedef unsigned int Uint;

using namespace std;
using namespace NGraph;


struct ND_pair    // node/degree pair
{
 Graph::const_iterator pnode;
 Uint degree;

};

std::ostream & operator << (std::ostream &s, const ND_pair &a)
{
  s << "(" << Graph::node(a.pnode) << "," << a.degree << ")";
  return s;
}

bool degree_order( const ND_pair &a, const ND_pair &b)
{
    return a.degree < b.degree; 
}

bool node_order( const ND_pair &a, const ND_pair &b)
{
    return Graph::node(a.pnode) < Graph::node(b.pnode); 
}

int main()
{

  Graph G;
  equivalence<Graph::vertex> E;


  cin >> G;
  vector<ND_pair> ND(G.num_vertices());

  // build node/degree array

  unsigned int i=0;
  for (Graph::const_iterator p=G.begin(); p!=G.end(); p++, i++)
  {
      ND_pair P = { p, Graph::in_degree(p) };
      //ND[i].pnode  = ND_pair(p, Graph::in_degree(p));
      //ND[i].degree = Graph::in_degree(p) ;
      ND[i] = P;
  }
   

  std::sort(ND.begin(), ND.end(), degree_order);
#if 0 
  for (unsigned int i=0; i< ND.size(); i++)
  {
    std::cout << ND[i] << "\n";
  }
  std::cout << "\n";
#endif

  while nodes still left in ND
  {
      // get next set of nodes with same degree
      D = list of nodes to process
      d = current degree value
      // for each node in its out degree
      for ( node_iterator v = D.begin(); v!=D.end(); v++)
      {
          const vertex_set &O = Graph::out_neighbors(v);
          for ( vertex_set::const_iterator p = O.begin(); p!=O.end(); p++)
          {
              if ( Graph::in_degree(p) <= d)
                E.insert(Graph::node(v), Graph::node(p);
          }
          const vertex_set &I = Graph::in_neighbors(v);
          for ( vertex_set::const_iterator p = I.begin(); p!=I.end(); p++)
          {
              if ( Graph::in_degree(p) <= d)
                E.insert(Graph::node(p), Graph::node(v);
          }
          E.insert(Graph::node(v));
      }
     
      std::cout << d << " " << E.num_classes() << " " << E.max_class_size() <<
              " " << E.num_elements() << "\n";
  } 

  return 0;
}


