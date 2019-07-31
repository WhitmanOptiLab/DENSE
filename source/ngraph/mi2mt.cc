#include <iostream>
#include <fstream>
#include <map>
#include <string>


/*

  Transform an integer-based community listing (.mi) into a URL-based (.mt)

  Usage :   cat foo.mi | mi2mt foo.t > foo.mt 

*/

typedef unsigned int Int;
using namespace std;

int main(int argc, char *argv[])
{

  if (argc <= 1)
  {
    std::cerr << "Usage :   cat foo.mi | mi2mt [-m foo.imap0] foo.t > foo.mt \n";
    exit(1);
  }


    // build URL map  (with 0-based remapping, if necessary)
    bool perform_remap = false;
    char *T_filename = 0;
    map<Int, Int> remap;
    string arg1(argv[1]);

    if (arg1 == "-m")
    {
      char *map_filename = argv[2];
      T_filename = argv[3];
      // build remap
      ifstream map_file;
      map_file.open(map_filename);
      while(map_file)
      {
        Int node1, node2;
        map_file >> node1 >> node2;
        remap[node1] = node2;
      }
      map_file.close();
      perform_remap = true;
    }
    else
    {
      T_filename = argv[1];
    }
    
    ifstream t_file;
    t_file.open(T_filename);
    
    map<Int, string> M;
    Int node = 0;
    string URL;

    while (t_file)
    {
      t_file >> node;
      getline(t_file, URL);
      size_t eat_white_space = URL.find_first_not_of(" \t");
      if ( eat_white_space != 0)
      {
         URL = URL.substr(eat_white_space);
      }
      if (perform_remap)
      {
        M[remap[node]] = URL;
      }
      else
      {
         M[node] = URL;
      }
    }
    t_file.close(); 


    // Now process the .mi file [node# community#]
    // .mt is in form [community#  URL ]
    //

    Int community_index = 0;
    while (cin)
    {
        cin >> community_index >> node ; 
        cout << community_index << " " << M[node] << "\n";
    }


    return 0;
    
}
