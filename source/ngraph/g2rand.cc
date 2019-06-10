// Converts a regular '*.g' file into a directed configuration model
//
// NOTE: Assumes graph is 0-based and contiguous
//

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <cmath>
#include <ctime>
#include "ngraph.hpp"
#include "tnt_stopwatch.h"
//
//  Usage g2rand < graph.g > graph_rand.g
//

using namespace std;
using namespace NGraph;

typedef unsigned int UInt;

typedef struct
{
    Graph::vertex from;
    Graph::vertex to;
}
Edge;

inline void randomize_init()
{
  srand((unsigned int) time(0) );
}

const double one_over_RAND_MAX = 1.0 / (RAND_MAX + 1.0);

inline UInt random(UInt low, UInt high)
{
    UInt range = (high-low)+1;
    return low + (UInt) ( range * rand() * one_over_RAND_MAX );

}


int main(int argc, char *argv[])
{
  bool print_timing = false;
  if (argc > 1)
  {
    std::string arg1(argv[1]);
    print_timing = (arg1 == "-t");
  }

  vector<Graph::vertex> V;
  vector<Edge> E;

  TNT::Stopwatch Q_total, Q_read, Q_rewire, Q_cleanup, Q_write;;
  

  Q_total.start();

  Q_read.start();
  // read in 0-based contiguous graph
  // (if not, then prefilter with g2g0)
  //
  std::istream &s = std::cin;
  std::string line;

    while (getline(s, line))
    {
      Graph::vertex v1, v2;

      if (line[0] == '%' || line[0] == '#')
        continue;

      std::istringstream L(line);
      L >> v1;
      if (L.eof())
      {
          V.push_back(v1);
      }
      else
      {
        L >> v2;
        Edge e = {v1, v2};
        E.push_back( e );
      }
    }
  Q_read.stop(); 
  if (print_timing)
  {
    cerr << "read time     : " << Q_read.read() << " secs\n";
  }
  



  Q_rewire.start();
  // now, for each edge in E, pick another random edge to swap nodes with
  //

  randomize_init();     // uses sytem time to generate different graphs
  unsigned int num_edges = E.size();
  for (UInt i=0; i<num_edges; i++)
  {
      // pick another edges at random
      UInt r = random(0, num_edges-1);
      swap( E[i].to, E[r].to );
  }
  
  Q_rewire.stop();
  if (print_timing)
  {
    cerr << "rewiring time : " << Q_rewire.read() << " secs\n";
  }
  

  Q_cleanup.start();
  // now convert to NGraph Graph to remove multiple edges
  // and self-loops

  Graph G;
  
  // build edges 
  for (UInt i=0; i<num_edges; i++)
  {
    UInt from = E[i].from;
    UInt to = E[i].to;

    if ( to != from)
      G.insert_edge( from, to );
  }
  
  // and any isolated nodes
  for (UInt i=0; i<V.size(); i++)
  {
    G.insert_vertex(V[i]);
  }

  Q_cleanup.stop();
  if (print_timing)
  {
    cerr << "cleanup time  : " << Q_cleanup.read() << " secs \n";
  }
 
  Q_write.start();
  cout << G;
  Q_write.stop();
  if (print_timing)
  {
    cerr << "write time    : " << Q_write.read() << " secs \n";
  }
 
  Q_total.stop();
  if (print_timing)
  {
    cerr << "total time    : " << Q_total.read() << " secs \n";
  }
 

}

