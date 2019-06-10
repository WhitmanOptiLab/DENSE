#include <iostream>
#include <fstream>
#include <map>
#include <string>


/*

  Transform a 0-based community listing (.mi0) back to original vertex numbers.
  Requires the original mapping of [original-vertex -> 0-based-vertex]

  Usage :   cat foo.mi0 | mi0_2mi foo.imap0 > foo.mi 

*/

typedef unsigned int Int;
using namespace std;

int main(int argc, char *argv[])
{

  if (argc <= 1)
  {
    std::cerr << "Usage :   cat foo.mi0 | mi0_2mi foo.imap0 > foo.mi\n";
    exit(1);
  }


    // build the invermap  map  
    map<Int, Int> inv_map;

      const char *map_filename = argv[1];
      ifstream map_file;
      map_file.open(map_filename);
      while(map_file)
      {
        Int node1, node2;
        map_file >> node1 >> node2;
        inv_map[node2] = node1;
      }
      map_file.close();
    
    
    // now start applying inverse map
    
    while (cin)
    {
        Int community_number;
        Int zero_based_vertex;
        cin >> community_number >> zero_based_vertex  ;
        cout << community_number << " " << inv_map[zero_based_vertex] << "\n";
    }


    return 0;
    
}
