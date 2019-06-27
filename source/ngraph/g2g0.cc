#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include "ngraph.hpp"

/*
    convert a general integer graph file into a contigous 0-based
    graph.

   -t option displays the table (mapping) rather than the graph

*/

using namespace std;
using namespace NGraph;

int main(int argc, char *argv[])
{

    if (argc ==2 && string(argv[1]) == "-h")
    {
        cerr << "Usage: " << argv[0] << " outfiename.g0 outfilename.imap0 \n";
        exit(1);
    }
   

    Graph G;

    cin >> G;

    typedef map<Graph::vertex, Graph::vertex> imap;
  
    imap M;

    for (Graph::const_iterator p = G.begin(); p!=G.end(); p++)
    {
      M[Graph::node(p)] = 0;
    }

    // now start numbering the nodes of G as 0,1,2, ...
    Graph::vertex count = 0;
    for (imap::iterator p = M.begin(); p!=M.end(); p++)
    {
        p->second = count++;
    }

      for (Graph::const_iterator p = G.begin(); p!=G.end(); p++)
      {
         Graph::vertex to = M[Graph::node(p)];
         for (Graph::const_vertex_iterator e=Graph::out_begin(p); 
                  e != Graph::out_end(p); e++)
         {
            cout << to << " " << M[Graph::node(e)] << "\n";
         }
      }

    return 0;
}
