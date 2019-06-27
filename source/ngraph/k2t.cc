/**

Take a pagerank file (.K) and a annotated-vertex file (.t) to produce 
an annotated list.

For example, if graph.v is

0
1
5
8
9

and graph.t looks like

0 http://math.nist.gov
1 http://gams.nist.gov/
2 http://www.nist.gov/itl/
3 http://www.itl.nist.gov/div898/projects/srm.htm
4 http://www.itl.nist.gov/div898/projects/keycomp.htm
5 http://math.nist.gov/mcsd/Seminars/
6 http://www.nist.gov/
7 http://www.itl.nist.gov/div898/consult/consult.html
8 http://math.nist.gov/workshops/witzgall-2004/
9 http://math.nist.gov/mcsd/savg/vis/
10 http://www.itl.nist.gov/div898/seminars/date.html

Then

cat graph.v | v2t graph.t


would produce

0 http://math.nist.gov
1 http://gams.nist.gov/
5 http://math.nist.gov/mcsd/Seminars/
8 http://math.nist.gov/workshops/witzgall-2004/
9 http://math.nist.gov/mcsd/savg/vis/

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
      s >> node_num >> label;
      M[node_num] = label;
  }


  while (getline(cin, line))
  {
      if (line.length() > 0)
      {
        stringstream s(line);
        uInt node_num;
        float pagerank;
        while (s>> node_num >> pagerank)
        {
          cout << node_num << " " << pagerank << " "<< M[node_num] << "\n";
        }
      }
      else
      {
        cout << "\n";
      }

  }


  return 0;
}

