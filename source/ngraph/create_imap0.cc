#include <iostream>
#include <set>

// Usage: create_imap [base]   (1-default)

using namespace std;

int main(int argc, char *argv[])
{
    typedef unsigned int Int;
    bool count_only = false;

    Int base = 0;
    
    if (argc > 1)
    {

       string option(argv[1]);

       if (option == "-n")
          count_only = true;
       else
          base = atoi(argv[1]);
    }

    set<Int> V;

    while (cin)
    {
      Int i;
      cin >> i;
      V.insert(i);
    }

    if (count_only)
    {
        cout << V.size() <<  "\n";
    }
    else
    {
      Int i=base;
      for (set<Int>::const_iterator p=V.begin(); p!=V.end(); p++, i++)
      {
        cout << *p << " " << i  << "\n";
      }
    }

    return 0;
}


