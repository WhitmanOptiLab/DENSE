//
// usage:  cat graphA.g graphB.g | gcompact  
//

#include <iostream>
#include <fstream>
#include "ngraph.hpp"


using namespace std;
using namespace NGraph;

int main(int argc, char *argv[])
{

   Graph A;
   cin >> A;
   cout << A;

   return 0;
}

