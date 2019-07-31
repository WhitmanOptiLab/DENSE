#include <iostream>
#include <fstream>
#include <map>
#include <string>


/*

  Extract a partial URL list, given an contiguous 0-based mapping:
  i.e. (.t) into a URL-based (.t0)

  Usage :   cat foo.t | t2t0 foo.imap0 > foo.t0 

*/

typedef unsigned int Int;
using namespace std;

int main(int argc, char *argv[])
{

  if (argc <= 1)
  {
    std::cerr << "Usage :   cat foo.t | t2t0 foo.imap0 > foo.t0\n";
    exit(1);
  }


    // read imap0 
    
    const char *imap_filename = argv[1];
    ifstream imap_file;
    imap_file.open(imap_filename);
    
    map<Int, Int> M;
    Int x=0, y=0;
    while (imap_file)
    {
      imap_file >> x >> y;
      M[x] = y;
    }
    imap_file.close();


    // now read in the URL's form the .t file and relabel them
    // if they are in M


    while (cin)
    {
      Int node;
      string URL;

      cin >> node;
      getline(cin, URL);
      map<Int, Int>::const_iterator p=M.find(node);
      if (p != M.end())
      {
        size_t eat_white_space = URL.find_first_not_of(" \t");
        if ( eat_white_space != 0)
        {
            URL = URL.substr(eat_white_space);
        }
        cout << p->second << " " << URL << "\n";
      }
    }


    return 0;
    
}
