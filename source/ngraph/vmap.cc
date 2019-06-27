/**

How is this different than v2t?

Take a vertex list (.v) and a node map and returns the labels associated with
each vertex number interger. For example, if foo.map looks like

0 0771044445
1 0827229534
2 0804215715
3 156101074X
4 0687023955

and foo.v lists

4
2
3


Then, 

  cat foo.v | vmap foo.map

creates

0687023955
0804215715
156101074X


Note that this operation is a text-based, that is each column is considered 
a separate text label. 


*/


#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <map>
#include <set>

using namespace std;

typedef map<unsigned int, string> node_array_type;

int main(int argc, char *argv[])
{

  typedef unsigned int uInt;


  ifstream t_file;
  t_file.open(argv[1]);
  if (!t_file) 
  {
    exit(1);
  }
  

  // read entire URL list into memory
  
  map<uInt, string> M;
  string line;
  while ( getline(t_file, line))
  {
      uInt node_num;
      string label;
      stringstream s(line);
      s >> node_num ;
      s >> label;
      M[node_num] = label;
  }


  while (getline(cin, line))
  {
      if (line.length() > 0)
      {
        stringstream s(line);
        uInt node_num;
        while (s>> node_num)
        {
          cout <<  M[node_num] << " ";
        }
        cout << "\n";
      }
      else
      {
        cout << "\n";
      }

  }


  return 0;
}

