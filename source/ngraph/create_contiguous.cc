
#include <set>
#include <iostream>

using namespace std;

int main()
{

    typedef unsigned int Integer;

    set<Integer> S;

    while (! cin.eof() )
    {
        Integer i;
        cin >> i;
        S.insert(i);
    }

    Integer I=0;
    for (set<Integer>::const_iterator p = S.begin(); p!=S.end(); p++, I++)
    {
        cout <<  *p << " " << I << "\n";
    }

    return 0;
}
      


