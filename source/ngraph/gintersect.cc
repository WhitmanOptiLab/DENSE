//
// usage:  cat graphA.g | gunion graphB.g > graph_uion.g
//

#include <iostream>
#include <fstream>
#include "ngraph.hpp"


using namespace std;
using namespace NGraph;

int main(int argc, char *argv[])
{


   if (argc < 2)
   {
     cerr << "Usage:  cat graphA.g | gunion graphB.g > graph_intersect.g\n";
     exit(1);
   }

   const char *graph_filename_B  = argv[1];
   ifstream graph_file_B;
   graph_file_B.open(graph_filename_B);

   Graph A;
   cin >> A;

   Graph B;
   graph_file_B >> B;
   graph_file_B.close();


   //cout << B.intersect(A);
   cout << (B * A);

   return 0;
}


