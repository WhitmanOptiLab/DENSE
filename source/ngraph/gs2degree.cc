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
  sGraph A;

  bool undirected = false;
  if (argv[1])
  {
    std::string arg1(argv[1]);
    undirected = (arg1 == "-u");
  }

  size_t max_string_len = 0;

  while (!std::cin.eof())
  {
     string v1, v2;

     std::cin >> v1 >> v2;

     max_string_len = max(v1.length(), max_string_len);
     max_string_len = max(v2.length(), max_string_len);
     A.insert_edge(v1, v2);
     if (undirected)
        A.insert_edge(v2, v1);
  }


   //unsigned int print_width =  static_cast<unsigned int>(
   //                       ceil(log10(A.num_nodes()) + 2));
   
  unsigned int print_width =  max_string_len+1;

  for ( sGraph::const_iterator p = A.begin(); p!=A.end(); p++)
  {
      std::cout 
                <<  left 
                << setw(print_width) 
                <<  sGraph::node(p) 
                << ' '
                ;

      if (undirected)
      {
         std::cout <<  sGraph::in_neighbors(p).size(); 
      }
      else
      {
         std::cout <<  sGraph::in_neighbors(p).size() 
            << ' '
            <<  sGraph::out_neighbors(p).size() ;
      }
      cout  << "\n";
      std::cout.flush();
  }

}

