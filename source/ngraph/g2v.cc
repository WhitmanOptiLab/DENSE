#include <iostream>
#include <set>
using namespace std;

int main()
{
    typedef unsigned int uInt;

    set<uInt> S;

    uInt node;

    while (cin>>node)
    {
        S.insert(node);
    }

    for (set<uInt>::const_iterator p=S.begin(); p!=S.end(); p++)
    {
       cout << *p << "\n";
    }
}
      
    
