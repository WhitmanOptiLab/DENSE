// gdd.cc creates a discrete degree distribution
//
//  (basically a word frequency)
//  and prints out the degree and how many times it occured

#include <iostream>
#include <string>
#include <map>

#define foreach(p,C)   for (p=C.begin(); p!=C.end(); (p)++)

using namespace std;

int main()
{
   map<string, unsigned int> F;

   while(cin)
   {
      string word;
      cin >> word;
      if ( word != "")
        F[word]++;
   }


  // now print out the frequency list

  map<string, unsigned int>::const_iterator p;
  foreach(p, F)
  {

    cout << p->first << "  " << p->second << "\n";
  }

  return 0;
}
