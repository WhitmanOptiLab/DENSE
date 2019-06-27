#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include "ngraph.hpp"

/*
    convert a general label (single-word string)graph file into a 
    contigous 0-based graph.

   -t option displays the table (mapping) rather than the graph

*/

using namespace std;
using namespace NGraph;

int main(int argc, char *argv[])
{
    bool perform_mapping_only = false;

    typedef string word;

    if (argc > 1)
      perform_mapping_only = ("-t" == string(argv[1]));
    
    Graph G;

             /*  old index     0-based index */
             /*  ---------     ------------- */
    typedef map<word, Graph::vertex> map_type;
    map_type M;
    Graph::vertex vertex_num = 0;

    std::string input_line;
    while (getline(std::cin, input_line))
    {
        word to, from;
        Graph::vertex to0, from0;
        stringstream s(input_line);
        s >> from >> to;

        //cout << "processed '" << from <<"' ->  '" << to << "'\n";


        map_type::iterator from_loc = M.find(from);
        if (from_loc == M.end())
        {
            from0 = M[from] = vertex_num;
            vertex_num++;

        }
        else
          from0 = from_loc->second;

        map_type::iterator to_loc = M.find(to);
        if (to_loc == M.end())
        {
            to0 = M[to] = vertex_num;
            vertex_num++;

        }
        else
          to0 = to_loc -> second;
            
            

        if (! perform_mapping_only)        
            G.insert_edge(from0, to0);
    }


   if (perform_mapping_only)
   {
     for (map_type::const_iterator p = M.begin(); p!= M.end(); p++)
     {
        cout <<  p->second << " " << p->first << "\n";
     }
   }

  else

    cout << G ;

  return 0;
}

