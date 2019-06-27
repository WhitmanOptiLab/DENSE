// Converts a regular '*.g' file into a line-separated adjancency file, 
// i.e. the node and each of its neighbors appear on different lines,
// with nodes separated by blank line.  For exmample, the regular 
// adjacency file
//
//  a  b c d
//  b  a e
//  c  d  f
//
//  where a is conneted to b, c, d,  and b is connected to a, e, and so on,
//  becomes
//
// a
// b
// c
// d
//
// b
// a
// e
//
// ...


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
      cout << Graph::node(p) << " :\n";  

      vertex_set E  = Graph::out_neighbors(p);

      for (vertex_set::const_iterator q = E.begin(); q!=E.end(); q++)
      {
        cout << *q << "\n";
      }
      cout << "\n";
  }

}

