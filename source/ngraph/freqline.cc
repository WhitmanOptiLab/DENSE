//  freqline.cc
//
//  (line frequency)

#include <iostream>
#include <string>
#include <map>

#define foreach(p,C)   for (p=C.begin(); p!=C.end(); (p)++)

using namespace std;

int main()
{
   map<string, unsigned int> F;

  string input_line;
   while( getline(cin, input_line))
   {
      F[input_line]++;
   }


  // now print out the frequency list

  map<string, unsigned int>::const_iterator p;
  foreach(p, F)
  {

    cout << p->first << "  " << p->second << "\n";
  }

  return 0;
}
