#include <iostream>
#include <string>
#include "token.hpp"

using namespace std;

int main()
{
  string line;

  while (getline(cin, line))
  {
      Tokenizer T(line, "\n\r");
      string word;

      while (! (word = T.next()).empty())
      {
          cout << "[" << word << "]\n";
      }
      cout << "\n";
  }
  return 0;
}


