#include <iostream>
#include <iostream>
#include <string>
#include "ngraph.hpp"
#include "pagerank.hpp"

// computes page_rank for a graph
//
//
//  Usage  cat graph.g |  gpagerank [num-iteratons] [min-delta] > graph.p    
//
// two colums: [node] [pagerank]
//

using namespace std;
using namespace NGraph;


typedef unsigned int uInt;

int main(int argc, char *argv[])
{
  
  uInt max_iterations = 200; 
  
  // stop if page rank values change by max of 0.0001%
  double max_delta = 0.000001;              

  if (argc >1)
    max_iterations = atoi(argv[1]);

  if (argc > 2)
    max_delta = atof(argv[2]);


  sGraph G;
  cin >> G;
 
  uInt iterations_used = 0;
  map<sGraph::vertex, double> P = pagerank(G, iterations_used, max_iterations, 
          max_delta);

  cerr << "iterations = " << iterations_used << 
          "  max_delta = " << max_delta << "\n";
  for (map<sGraph::vertex, double>::const_iterator i=P.begin(); 
                i!=P.end(); i++)
  {
      cout << i->first << "  " << i->second << "\n";
  }


  return 0;
}

