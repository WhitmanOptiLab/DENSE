// Create an Erdos-Renyi random graph, given a specific number of nodes
// and edges.


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
//  Usage:  gg_erdos_renyi 100 150 > rand_V100_E150.g 
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
  if (argc < 3)
  {
    cerr << "Usage:  " << argv[0] << " num_vertices   num_edges \n";
    exit(1);
  }

  UInt num_nodes = atoi(argv[1]);
  UInt num_edges = atoi(argv[2]);

  randomize_init();     // uses sytem time to generate different graphs

  for (UInt i=0; i< num_edges; i++)
  {
      UInt a = random(0, num_nodes-1);
      UInt b = random(0, num_nodes-1);
      cout << a << " " << b << "\n";
  }

  return 0;
}


