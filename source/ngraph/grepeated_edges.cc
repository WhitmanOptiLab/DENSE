#include <iostream>
#include "ngraph.hpp"


using namespace std;
using namespace NGraph;

int main()
{
   Graph G;
   bool vertex_only = false;
   Graph::vertex v1=0, v2=0;

   while ( Graph::read_line(cin, v1, v2, vertex_only))
   {
          if (vertex_only)
          {
            G.insert_vertex(v1);
          }
          else
          {
              if (G.includes_edge(v1, v2))
                cout << v1 << " " << v2 << "\n";
              else
                G.insert_edge(v1,v2);
          }
    }

    return 0;
}


