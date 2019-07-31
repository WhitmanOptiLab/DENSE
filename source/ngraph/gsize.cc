#include <iostream>
#include "ngraph.hpp"


using namespace std;
using namespace NGraph;

int main()
{
   Graph G;

   cin >> G;
   cout << G.num_vertices() << " " << G.num_edges() << "\n";
   //cout << G;

   return 0;
}
