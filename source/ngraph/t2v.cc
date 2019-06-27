/**

Takes a list  urls  and returns their vertex number, using the corresponding .t file.

For example, if url list is

http://math.nist.gov
http://gams.nist.gov/
ttp://math.nist.gov/mcsd/savg/vis/
http://math.nist.gov/mcsd/Seminars/
http://math.nist.gov/workshops/witzgall-2004/

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

cat url.txt | t2v graph.t


would produce

0 
1 
5 
8 
9

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
  
  map<string, uInt> M;
  string line;
  while ( getline(t_file, line))
  {
      uInt node_num;
      string label;
      stringstream s(line);
      s >> node_num >> label;
      M[label] = node_num;
  }


  while (getline(cin, line))
  {
      if (line.length() > 0)
      {
        stringstream s(line);
        string label;
        while (s>> label)
        {
          cout <<  M[label] << "\n";
        }
      }
      else
      {
        cout << "\n";
      }

  }


  return 0;
}

