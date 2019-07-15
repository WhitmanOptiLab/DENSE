#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <cmath>
#include "ngraph.hpp"

//
//  Usage a.out [-u] < graph.dat     (-u for undirected)
//

using namespace std;
using namespace NGraph;

int main(int argc, char *argv[])
{
  Graph A;

  bool undirected = false;
  if (argv[1])
  {
    std::string arg1(argv[1]);
    undirected = (arg1 == "-u");
  }

  cin >> A;

  for ( Graph::const_iterator p = A.begin(); p!=A.end(); p++)
  {
      std::cout <<  std::left << " "  <<  Graph::node(p) ;

      if (undirected)
      {
         std::cout <<  " " << Graph::in_degree(p); 
      }
      else
      {
         std::cout <<  " " << Graph::in_degree(p)
            << " " <<  Graph::out_degree(p) ;
      }
      cout  << "\n";
      std::cout.flush();
  }

}

