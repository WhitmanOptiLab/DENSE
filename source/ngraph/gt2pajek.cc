/*
   Given a grah file (.g) and a (larger) annotated vertex file (.t),
   created a directed Pajek file (.net) .

   Usage:  gt2pajek graph.g graph.t > graph.net


   Short exmaple of Pajek file format:

-------------------------------------
*Vertices 3
1 "Doc1"
2 "Doc2"
3 "Doc3"
*Arcs
1 2
2 3
-------------------------------------



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

   if (argc < 3)
   {
       cerr << "Usage: " << argv[0] << " graph.g  graph.t \n";
       exit (1);
   }

   const char *graph_filename = argv[1];
   const char *vertex_label_filename = argv[2];
   ifstream graph_file;
   ifstream vertex_label_file;

   //cerr << "Opening ["<< graph_filename << "]" << endl;

   graph_file.open( graph_filename);
   if (!graph_filename)
   {
       cerr << "Error: [" << graph_filename << "] could not be opened.\n";
       exit(1);
   }


   vertex_label_file.open(vertex_label_filename);
   if (!vertex_label_filename)
   {        cerr << "Error: ["<< vertex_label_filename<< "] could not
be opened.\n";
       exit(1);
   }




  Graph G;
  graph_file >> G;

  map<Graph::vertex, string> T;

   while (!vertex_label_file.eof())
   {
       Graph::vertex node;

       vertex_file >> node ;
       V.insert(node);
   }

  Graph G;
  graph_file >> G;

  Graph G_V = G.subgraph(V);

   cout << G_V;

   return 0;
