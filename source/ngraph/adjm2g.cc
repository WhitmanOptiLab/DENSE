/* This converts an multi-line adjancy list graph to an edge list.  The format
   is, each node is listed with its neighbors, each on a separate line.  The
   file looks, then, like a group of blank-line separated nodes, with the
   first node in each group being the root node, and those following it are
   its neighbors. For example,
  
   a 
   b 
   c 

   d
   c
   e

   is the same as the (single line) adjacency format:

    a: b c
    d: c e
 
   and the edge-list format

   a b
   a c
   d c
   d e

*/  

#include <iostream>
#include <string>
#include "token.hpp"

using namespace std;



int main()
{

	
  std::string word;
  std::string input_line;

  while (getline(std::cin, input_line))
  {
      if (input_line.length() < 1)
          continue;

      string home_node = input_line;
      while (getline(std::cin, input_line))
      {
          if (input_line.length() < 1)
             break;

          std::cout << home_node << " " << input_line << "\n";
      }
  }


  return 0;
}

