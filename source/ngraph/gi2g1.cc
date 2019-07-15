#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include "ngraph.hpp"

/*
    convert a general integer graph file into a contigous 1-based
    graph.

   -t option displays the table (mapping) rather than the graph

*/

using namespace std;
using namespace NGraph;

int main(int argc, char *argv[])
{
    bool perform_mapping_only = false;

    if (argc > 1)
      perform_mapping_only = ("-t" == string(argv[1]));
    
    Graph G;

    typedef Graph::vertex input_graph_type;
    typedef map<input_graph_type, Graph::vertex> map_type;
    map_type M;
    Graph::vertex vertex_num = 0;

    std::string input_line;
    while (getline(std::cin, input_line))
    {
        input_graph_type to, from;
        stringstream s(input_line);
        s >> from >> to;

        //cout << "processed '" << from <<"' ->  '" << to << "'\n";

        Graph::vertex ifrom = M[from];
        if (ifrom == 0)
        {
            vertex_num++;
            ifrom = M[from] = vertex_num;
        }
            
        int ito = M[to];
        if (ito == 0)
        {
            vertex_num++;
            ito = M[to] = vertex_num;
        }
            

        if (! perform_mapping_only)        
            G.insert_edge(ifrom, ito);
    }


   if (perform_mapping_only)
   {
     for (map_type::const_iterator p = M.begin(); p!= M.end(); p++)
     {
        cout <<  p->first << " " << p->second << "\n";
     }
   }

  else

    cout << G ;

  return 0;
}

