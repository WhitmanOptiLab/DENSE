// Converts a regular '*.g' file into an adjancency file, each line contains
// a vertex and its outgoing neighbors, e.g.
//
//  a  b c d
//  b  a e
//  c  d  f
//


#include <iostream>
#include "ngraph.hpp"

//
//  Usage a.out < graph.g    
//

using namespace std;
using namespace NGraph;

typedef Graph::const_iterator const_node_iterator;
typedef Graph::vertex_set  vertex_set;

int main(int argc, char *argv[])
{


  Graph A;

  cin >> A;
  
  for (const_node_iterator p=A.begin(); p!=A.end(); p++)
  {

      vertex_set E  = Graph::out_neighbors(p);

      if (E.size() > 0)
      {
        cout << Graph::node(p) << " ";  
        for (vertex_set::const_iterator q = E.begin(); q!=E.end(); q++)
        {
           cout << *q << " ";
        }
        cout << "\n";
      }
  }

}

