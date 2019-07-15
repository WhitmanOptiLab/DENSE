/*
    Given a grah file (.g) from stdin, create a sugraph of only those nodes listed in a vertex file (.v)

    Usage:  zcat math.g.gz | greduce math.v

*/


#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include "ngraph.hpp"
#include "set_ops.hpp"

using namespace std;
using namespace NGraph;


int main(int argc, char *argv[])
{
  
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " file.v \n";
        exit (1);
    }

    const char *vertex_filename = argv[1];
    
    ifstream vertex_file;
    vertex_file.open( vertex_filename);
    
    //cerr << "Opening ["<< vertex_filename << "]" << endl;

    if (!vertex_filename)
    {
        cerr << "Error: [" << vertex_filename << "] could not be opened.\n";
        exit(1);
    }

    Graph<string> G;
    set<string> V;

    // read in vertices and build set
    while (!vertex_file.eof())
    {
        string node_num;

        vertex_file >> node_num ;

        if (node_num !="")
        {
          //cerr << "read " << node_num << endl;
          V.insert(node_num);
        }
    }
    vertex_file.close();

    while (!std::cin.eof())
    {
       string to, from;
       std::cin >> to >> from;
       if (includes_elm(V, to) && includes_elm(V, from))
       G.insert_edge(to, from);
    }

    cout << G;


    return 0;
}
