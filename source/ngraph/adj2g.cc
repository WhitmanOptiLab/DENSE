/* This converts an adjancy list graph to an edge list.  The format
   is 
  
   node-num  a b c 

    where a, b, c, etc. are the neighbors.  One can use ':' or ',' as
    delmiters, i.e.

    node-num    a b c
    node-num :  a b c
    node-num :  a, b, c
  
   are all the same.
*/  

#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <list>
#include "token.hpp"

using namespace std;



int main()
{

	
  const string delimiters = ",: \t";       // white space
  std::string input_line;

  // read past the first two lines
  //

  while (getline(std::cin, input_line))
  {
	    Tokenizer T(input_line, delimiters);
      string from_node = T.next();
      string to_node = T.next();
      if (to_node.empty())
      {
         std::cout << from_node << "\n";
      }
      else
      {
         std::cout << from_node << " " << to_node << "\n";
      }
      std::string neighbor;
	    while ( ! (neighbor=T.next()).empty())
        {
          std::cout << from_node << " " << neighbor << "\n";
        }
  };

  return 0;
}

