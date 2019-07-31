// Converts a regular '*.g' file into a directed configuration model
// by carefully selecting "non-local" nodes to swap edges with.
//
// NOTE: This is a modificaiton of g2prand which has an extra step to
//       select two edges (e and r) to switch in which e.to is not part
//       of r.from neighborhood and r.to is not part of e.from neighborhood.
//     
//       Because we need a fast way to check for neighborhoods, we need to
//       build an auxillary graph G, which takes up about double the memory.
//
//       Like  g2prand, it assumes the graph is 0-based and contiguous
//

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <cmath>
#include <ctime>
#include "ngraph.hpp"
#include "tnt_stopwatch.h"
#include "set_ops.hpp"

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

const double one_over_RAND_MAX = 1.0 / (double) (RAND_MAX);

inline UInt random(UInt low, UInt high)
{
    UInt range = (high-low)+1;
    return low + (UInt) ( (range * one_over_RAND_MAX)  * rand() ) ;

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
  Graph G;

  TNT::Stopwatch Q_total, Q_read, Q_rewire, Q_write;;
  

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
          G.insert_vertex(v1);
      }
      else
      {
        L >> v2;
        Edge e = {v1, v2};
        E.push_back( e );
        G.insert_edge(v1, v2);
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

  unsigned int num_edges = E.size();
  randomize_init();     // uses sytem time to generate different graphs
  
  UInt min_r = random(0, num_edges-1);
  UInt max_r = min_r;
  UInt num_resampled_nodes = 0;
  UInt total_num_of_resamples = 0;

  UInt max_resampled_nodes = num_edges * 0.95;
  
  for (UInt i=0; i<num_edges; i++)
  {
      // pick another edge at random
      UInt r = random(0, num_edges-1);

      UInt num_resamples = 0;

      UInt a = E[i].from;
      UInt b = E[i].to;
      UInt c = E[r].from;
      UInt d = E[r].to;

      // now keep re-sampling, if needed, until we find two edges
      // whose neighborhoods don't overlap, and don't from self-loops
      //
      while (a == d || c == b || G.includes_edge(a,d) || G.includes_edge(c,b) )
      {
          r  = random(0, num_edges-1);
          c = E[r].from;
          d = E[r].to;
          
          num_resamples++;
          if (num_resamples > max_resampled_nodes)
          {
              cerr << "Error: too many edges resampled: edge #" << i 
                << " : (a,b) = ("
                << a << ", " << b <<") ; " << num_resamples 
                << " / " << num_edges << ".\n";
              exit(1);
          }
      }

      if (r < min_r)
        min_r = r;
      else if (r > max_r)
        max_r = r;

#if 0
      swap( E[i].to, E[r].to );
      G.remove_edge(E[i].from, E[i].to);
      G.insert_edge(E[i].from, E[r].to);
      G.remove_edge(E[r].from, E[r].to);
      G.insert_edge(E[r].from, E[i].to);
#endif

      E[i].to = d;
      E[r].to = b;

      G.remove_edge(a,b);
      G.insert_edge(a,d);

      G.remove_edge(c,d);
      G.insert_edge(c,b);

      if (num_resamples > 0)
      {
        num_resampled_nodes++;
        total_num_of_resamples += num_resamples;
      }

  }
  
  Q_rewire.stop();
  if (print_timing)
  {
    cerr << "rewiring time : " << Q_rewire.read() << " secs\n";
  }
  

  // now print out edes and vertices

  Q_write.start();
  
  // first, write out isolated node, if any
  for (UInt i=0; i<V.size(); i++)
  {
    cout << V[i] << "\n";
  }

  // then print edges
  for (UInt i=0; i<num_edges; i++)
  {

    cout << E[i].from << " " << E[i].to << "\n";
  }
  
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
 

  if (print_timing)
  {
    cerr << "min r: " << min_r << "\n";
    cerr << "max r: " << max_r << "\n";
    cerr << "num_edges :" << num_edges << "\n";
    cerr << "number of resampled nodes: " << num_resampled_nodes << "\n";
    cerr << "average number of resamples "<< 
      ((double) total_num_of_resamples) / num_resampled_nodes << "\n";
  }
}

