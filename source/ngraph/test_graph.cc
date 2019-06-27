#include "ngraph.hpp"


using namespace NGraph;
using namespace std;

int  main(int argc, char *argv[])
{
    if (argc < 3)
    {
      cout << "Usage:  a.out a b.\n";
      return 0;
    }
 
    Graph::vertex a = atoi(argv[1]);
    Graph::vertex b = atoi(argv[2]);

    Graph A;
    cin >> A;

    A.absorb(a, b);

    cout << A;

    return 0;

}
