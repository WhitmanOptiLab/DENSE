#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include "ngraph.hpp"

/*
    convert a general integer graph file into a contigous 0-based
    graph.

   -t option generates both graph and mapping as separate files, i.e.

   cat foo.g | g2g0 -t foo.g0 foo.imap

*/

using namespace std;
using namespace NGraph;

int main(int argc, char *argv[])
{

    if (argc <=2)
    {
        cerr << "Usage: " << argv[0] << " outfiename.g0 outfilename.imap0 \n";
        exit(1);
    }
   
    const string outfilename_g(argv[1]);
    const string outfilename_imap(argv[2]);

    Graph G;

    cin >> G;

    typedef map<Graph::vertex, Graph::vertex> imap;
  
    imap M;

    for (Graph::node_iterator p = G.begin(); p!=G.end(); p++)
    {
      M[Graph::node(p)] = 0;
    }

    // now start numbering the nodes of G as 0,1,2, ...
    Graph::vertex count = 0;
    for (imap::iterator p = M.begin(); p!=M.end(); p++)
    {
        p->second = count++;
    }


    // write mapping into .t0 file:  [ i   f(i) ]
    ofstream outfile_imap;
    outfile_imap.open(&outfilename_imap[0]);
        for (imap::iterator p = M.begin(); p!=M.end(); p++)
        {
          outfile_imap << p->first << " " << p->second << "\n";
        }
    outfile_imap.close();

    std::ofstream outfile_g;
    outfile_g.open(&outfilename_g[0]);
      for (Graph::node_iterator p = G.begin(); p!=G.end(); p++)
      {
         Graph::vertex to = M[Graph::node(p)];
         const Graph::vertex_set &E = Graph::out_neighbors(p);
         for (Graph::vertex_set::const_iterator e = E.begin(); e!=E.end();e++)
         {
            outfile_g << to << " " << M[*e] << "\n";
         }
      }
    outfile_g.close();

    return 0;
}
