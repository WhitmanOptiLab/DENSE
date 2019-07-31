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
  typedef int file_content_type;
  typedef  map<file_content_type, unsigned int> Word_count;

   Word_count F;

   file_content_type word;

   while(cin >> word)
   {
        F[word]++;
   }


  // now print out the frequency list

  Word_count::const_iterator p;
  foreach(p, F)
  {

    cout << p->first << "  " << p->second << "\n";
  }

  return 0;
}
