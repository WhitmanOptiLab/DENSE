// Reverses directions of edges in a directed graph.


#include <iostream>
#include "ngraph.hpp"
//
//  Usage a.out < graph.g    
//

//using namespace std;
using namespace NGraph;

typedef Graph::const_vertex_iterator cvit;

int main(int argc, char *argv[])
{

  Graph G;

  std::cin >> G;

  for ( Graph::const_iterator p = G.begin(); p!=G.end(); p++)
  {
     Graph::vertex v = Graph::node(p);
     if ( Graph::isolated(p))
     {
        std::cout << v << "\n";
     }
     else
     {
        for (cvit e = Graph::out_begin(p); e!=Graph::out_end(p); e++)
        {
          // for each (v,e) write (e,v)
          std::cout << Graph::node(e) << " " << v << "\n";
        }
     }
  }

  
  return 0;
}

