#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <set>
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

  typedef map< unsigned int, set<Graph::vertex> > DegreeList;
  DegreeList D;


  for ( Graph::const_iterator p = A.begin(); p!=A.end(); p++)
  {
      D[Graph::in_degree(p) + Graph::out_degree(p)].insert(Graph::node(p));
  }

  for (DegreeList::iterator d=D.begin(); d!=D.end(); d++)
  {
    cout << d->first;
    set<Graph::vertex> &S = d->second;
    for (set<Graph::vertex>::const_iterator v=S.begin(); v!=S.end(); v++)
    {
      cout << " " << *v ;
    }
    cout << "\n";
  }

}

