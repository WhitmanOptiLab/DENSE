/* This converts Brian's funky graph format into a simple edge list.  Basic
   format is an adjacency list:
  
    #node-num <junk> <junk>  : a : b : c : d

    where a, b, c, etc. are the neighbors
   
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

	
  const string delimiters = ": \t";       // white space
  std::string input_line;

  while (getline(std::cin, input_line))
  {
	      Tokenizer T(input_line, delimiters);
        string node_num = T.next();
        T.next();
        T.next();
        std::string neighbor;
	      while ( ! (neighbor=T.next()).empty())
        {
          std::cout << node_num << " " << neighbor << "\n";
        }
  };

  return 0;
}

