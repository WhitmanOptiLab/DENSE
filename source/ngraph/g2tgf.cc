/**

Convert a .g and .t file into an annotated Ttrivial Graphics Format (TGF).

usage:   cat graph.g | gt2tgf graph.t > graph.tgf

The graph.t file is ain optional list of node-lables.  For example, the 
TGF format of the graph V={1,2,3}, E = { (1, 3) } is given by

----
v1 label1
v2 label2
v3 label3
#
v1 v3
----

where v1, v2, v3 are integers and labels are strings.  The nodes (with 
optonal labels) are listed first, followed by a '#', then edge list.  
*/


#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <map>
#include <set>
#include "ngraph.hpp"

using namespace std;
using namespace NGraph;

typedef map<unsigned int, string> node_label_array;
typedef Graph::const_iterator const_node_iterator;
typedef Graph::vertex_set  vertex_set;

int main(int argc, char *argv[])
{

  typedef unsigned int uInt;
  bool node_labels  = false;
  node_label_array M;

  if (argc > 1)
    node_labels = true;

  if (node_labels)
  {
      ifstream t_file;
      t_file.open(argv[1]);
      if (!t_file) 
      {
        exit(1);
      }
  
		
		  // read entire node label file (.t) into memory
		  
		  string line;
		  while ( getline(t_file, line))
		  {
		      uInt node_num;
		      string label;
		      stringstream s(line);
		      s >> node_num >> label;
		      M[node_num] = label;
		  }
  }
		

  Graph G;

  cin >> G;

  if (node_labels)
  {
		 for (const_node_iterator p=G.begin(); p!=G.end(); p++)
		  {
		      Graph::vertex v = Graph::node(p);
		      node_label_array::const_iterator L = M.find(v);
		      if ( L != M.end())
		          cout << v << " " << L->second  << "\n";
		      else
		          cout << v <<  "\n";
		  }
	}
  else
  {
		 for (const_node_iterator p=G.begin(); p!=G.end(); p++)
		  {
		      Graph::vertex v = Graph::node(p);
		       cout << v <<  " " << v << "\n";
		  }
  }

  cout << "#\n";

#if 0
  for (const_node_iterator p=G.begin(); p!=G.end(); p++)
  {
      Graph::vertex this_node = Graph::node(p);
      vertex_set E  = Graph::out_neighbors(p);

      if (E.size() > 0)
      {
        for (vertex_set::const_iterator q = E.begin(); q!=E.end(); q++)
        {
           cout << this_node << " " << *q << "\n";
        }
      }
  }
#endif

  // modified version of  cout << G, where we don't print out 
  // isolated nodes.  (These are already accounted for in the TGF
  // node delcaration portion of the output file.

  //cout << G;

  for (Graph::const_node_iterator p=G.begin(); p != G.end(); p++)
  {
    const Graph::vertex_set &out = Graph::out_neighbors(p);
    Graph::vertex v = p->first;

    // if not isolated node..
    //
    if (!(out.size() == 0 && Graph::in_neighbors(p).size() == 0))
    {
       for (Graph::vertex_set::const_iterator q=out.begin();
                q!=out.end(); q++)
           cout << v << " " << *q << "\n";
    }
  }

  return 0;
}

