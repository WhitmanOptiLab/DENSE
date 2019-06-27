// This is a multigraph version of g2degree that allows for self-loops
// and multiple edges.  (This is required because the configuration model
// sometimes yields graphs which are not simple.)

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <cmath>
#include "ngraph.hpp"

//
//  Usage  cat foo.mg | mg2degree     
//

using namespace std;
using namespace NGraph;

typedef unsigned int UInt;
typedef struct
{
    UInt in_degree;
    UInt out_degree;
} in_out_degree;

typedef  map<Graph::vertex, in_out_degree> degree_tally;


int main(int argc, char *argv[])
{
  Graph A;

  bool undirected = false;
  if (argv[1])
  {
    std::string arg1(argv[1]);
    undirected = (arg1 == "-u");
  }


 // read in the graph and keep a running count of in and out degree


  degree_tally D;

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
          D[v1];
      }
      else
      {
        L >> v2;
        D[v1].out_degree++;
        D[v2].in_degree++; 
      }
    }

    for (degree_tally::const_iterator p=D.begin(); p!=D.end(); p++)
    {
      cout << p->first << " " << p->second.in_degree << " " 
          << p->second.out_degree << "\n";
    }

}

