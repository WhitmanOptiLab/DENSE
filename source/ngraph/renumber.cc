// Given a one-to-one mapping of integers (list of (i,j) pairs, map every i
// into j from standard input.
//
//  It is assumed the the input file ONLY contains (posivite) integers.
//
//  Note: stringstream is needed so that the program respects newlines.  
//        That is, it can distinguish between .g and .t files.
//

#include <string>
#include <set>
#include <map>
#include <iostream>
#include <sstream>
#include <fstream>

using namespace std;

int main(int argc, char *argv[])
{
    typedef unsigned int Int;

    if (argc <= 1)
    {
        cerr << "Usage: map_filename " << "\n";
    }

    const char *filename = argv[1];


    map<Int, Int> M;

    ifstream F;
    F.open(filename);
    Int from, to;
    while (F)
    {
        F >> from >> to;
        M[from] = to;
    }
    F.close();

    string line;
    while (getline(cin,line))
    {
        stringstream s(line);
        while (!s.eof())
        {
            Int i;
            s >> i;
            cout << M[i] << " ";
        }
        cout << "\n";
    }


    return 0;
}
      


