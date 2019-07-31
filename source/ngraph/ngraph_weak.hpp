#include <set>
#include "set_ops.hpp"
#include "ngraph.hpp"
#include "equivalence.hpp"

using namespace NGraph;

//
//  G is the original (complete) graph
//  E is the connected components (equivalance classes)
//  C is the accumulated nodes in all connected components of E 
//  begin-end is the list of new nodes being added

template <typename Node_Iter, typename Equivalence >
void weak_components_increment( const Graph &G, Equivalence &E,  
 std::set<Graph::vertex> &C,  Node_Iter begin, Node_Iter end)
{
  C.insert(begin, end);
  for (Node_Iter pv=begin; pv!=end; pv++)
  {
      Graph::vertex v = *pv;
        
       

        //cerr << "weak_components_increment: inserted (" << v << ")\n";
        Graph::const_iterator gv = G.find(v);
        for (Graph::const_vertex_iterator p=G.in_begin(gv); 
                  p!=G.in_end(gv); p++)
        {
          Graph::vertex e = Graph::node(p);
          if (includes_elm(C, e))
                E.insert(e, v);
        }
        for (Graph::const_vertex_iterator p=G.out_begin(gv); 
                  p!=G.out_end(gv); p++)
        {
          Graph::vertex e = Graph::node(p);
          if (includes_elm(C, e))
              E.insert(v, e);
        }

        // just in case there was no intersection with neighbors of v
        // and the nodes already in E.  We still need to record node v in E.
        //
        E.insert(v);
  }
}
