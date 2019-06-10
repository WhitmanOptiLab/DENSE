#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <cmath>
#include "set_ops.hpp"
#include "ngraph.hpp"

// removes terminal branches from graph 
//
// only works for directed graphs at the moment
//
//  Usage  cat foo.g |  gprune [-u] > foo_pruned.g
//

using namespace std;
using namespace NGraph;


const Graph::vertex & sole_in_neighbor(const Graph &A, const Graph::vertex &v )
{
    return   *(A.in_neighbors(v).begin());
}


const Graph::vertex & sole_in_neighbor( Graph::const_iterator p )
{
    return    *(Graph::in_neighbors(p).begin());

}

inline bool terminal_node(Graph::const_iterator p)
{
    return (Graph::out_neighbors(p).size() == 0 &&
            Graph::in_neighbors(p).size()  != 0    );
}

inline bool reflexive_node(Graph::const_iterator p)
{
    return (Graph::in_neighbors(p) == Graph::out_neighbors(p));
}

inline bool u_turn_node(Graph::const_iterator p)
{
    const Graph::vertex_set &in = Graph::in_neighbors(p);
    const Graph::vertex_set &out = Graph::out_neighbors(p);

    return  (in.size() == 1) && 
            (out.size() ==1) && 
            (*in.begin() == *out.begin()) ;
}




bool single_incoming_edge(Graph::const_iterator p)
{
    return (Graph::in_neighbors(p).size() == 1);
}


int main(int argc, char *argv[])
{
  Graph A;

  cin >> A;


#if 0
  //cout << "%% Original graph: (V,E) = (" << A.num_vertices() << ", " <<
  //       A.num_edges() << ")\n";
  //cout << A << "\n";
#endif

  // now we make multiple passes until we can no longer coarsen the
  // graph.


  set<Graph::vertex> D;     // nodes to delete
  set<Graph::vertex> C;     // nodes to check

  // first time through the loop to prime D and C
  //
  for (Graph::const_iterator p =A.begin(); p != A.end(); p++)
  {
      if ( terminal_node(p) )
      {
        //std::cerr << "found terminal node: " << Graph::node(p) << "\n";
        D.insert(Graph::node(p));
        C.insert( sole_in_neighbor(p) );
      }
      else if ( u_turn_node(p) )
      {
        //std::cerr << "found u-turn  node: " << Graph::node(p) << "\n";
        D.insert(Graph::node(p));
        C.insert( Graph::in_neighbors(p).begin(), 
                  Graph::in_neighbors(p).end());
      }
  }
  //std::cerr << "\n";

  //std::cerr << "|D| = " << D.size() << "\n";
  //std::cerr << "|C| = " << C.size() << "\n";

  while (D.size() > 0 )
  {
    //std::cerr << "*" ;

    typedef set<Graph::vertex>::const_iterator vertex_set_iterator;

    set<Graph::vertex> D1;     // nodes to delete
    set<Graph::vertex> C1;     // nodes to check

    for ( vertex_set_iterator v=D.begin(); v!=D.end(); v++)
    {
        //std::cerr << "About to remove vertex: " << *v << "\n";
        A.remove_vertex(*v);
    }


   // std::cerr << "Now going to check pruned branches: \n";
    
    for ( vertex_set_iterator v=C.begin(); v!=C.end(); v++)
    {
      Graph::const_iterator  p = A.find(*v);
      
      if (p != A.end())
      {
         if ( terminal_node(p) )
         {
           //std::cerr << "found terminal node: " << Graph::node(p) << "\n";
           D1.insert(Graph::node(p));
           C1.insert( sole_in_neighbor(p) );
         }
         else if ( u_turn_node(p) )
         {
           //std::cerr << "found u-turn  node: " << Graph::node(p) << "\n";
           D1.insert(Graph::node(p));
           C1.insert( Graph::in_neighbors(p).begin(), 
                     Graph::in_neighbors(p).end());
         }
      }
    }

    D = D1;
    C = C1;
  
    //std::cerr << "|D| = " << D.size() << "\n";
    //std::cerr << "|C| = " << C.size() << "\n";
  }
    
  // std::cerr << "\n";

#if 0
  cout << endl << "%% Reduced graph: (V,E) = (" << A.num_vertices() << ", " <<
          A.num_edges() << ")\n";
#endif

  cout << A ;
  return 0;
}

