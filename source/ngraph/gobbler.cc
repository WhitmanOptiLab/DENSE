#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <cmath>
#include "set_ops.hpp"
#include "ngraph.hpp"

//
//  Usage  cat foo.g |  gcoarsen [-u] > reduced_foo.g
//

using namespace std;
using namespace NGraph;


const Graph::vertex & first_in_neighbor(const Graph &A, const Graph::vertex &v )
{
    return   *(A.in_neighbors(v).begin());
}


const Graph::vertex & first_in_neighbor( Graph::const_iterator p )
{
    return    *(Graph::in_neighbors(p).begin());

}


int main(int argc, char *argv[])
{
  Graph A;

  bool undirected = false;
  if (argv[1])
  {
    std::string arg1(argv[1]);
    undirected = (arg1 == "-u");
  }


  while (!std::cin.eof())
  {
     Graph::vertex v1, v2;

     std::cin >> v1 >> v2;

     A.insert_edge(v1, v2);
     if (undirected)
        A.insert_edge(v2, v1);
  }


  cout << "# Original graph: (V,E) = (" << A.num_vertices() << ", " <<
          A.num_edges() << ")\n";
  //cout << A << "\n";

  // now we make multiple passes until we can no longer coarsen the
  // graph.



  while (1)
  {

    set<Graph::vertex> D;  // nodes which  have just one in-coming edge

    // find nodes with just one incoming edge


    for (Graph::iterator p = A.begin(); p !=A.end(); p++)
    {
       if ( Graph::in_neighbors(p).size() == 1 )
       {
          D.insert(
              Graph::node(p));
          std::cout <<"["<< first_in_neighbor(p) << ",  " << 
              Graph::node(p) << "]\n";
       }
    }

    if (D.size() < 1) break;


    // now remove (absorb) the nodes
    for (set<Graph::vertex>::const_iterator s = D.begin(); s != D.end(); s++)
    {
       Graph::vertex a = first_in_neighbor(A, *s);

       //cout << "about to absorb (" << a << ", " << *s << ")\n";
       A.absorb( a, *s );
       //cout << "absorbed (" << a << ", " << *s << ")\n";
    }

  }


  cout << endl << "# Reduced graph: (V,E) = (" << A.num_vertices() << ", " <<
          A.num_edges() << ")\n";
  cout << endl << A << endl;
  return 0;
}

