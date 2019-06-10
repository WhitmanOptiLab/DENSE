/*

   This filter is for processing discrete graphs for 3D visualization, using 
   the SAVG system at NIST.

   Takes a 3D vertex coordinate file (.crd) and a graph file (.g) and 
   produces a SAVE (.savg) file for interactive visualization.

   vertex coordinate file lists the 3d position of each node, and is in 
   the format of [#node-num x y z].  For example,

   1 x1 y1 z1
   2 x2 y2 z2
   3 x3 y3 z3
   ...

   The *.g file is a just an edge list (i,j).


   The output is a file in the format:

   lines
   xi yi zi 1 1 1 1
   xj yj zj 1 1 1 1

   for each edge (i,j)

*/


#include <iostream>
#include <fstream>
#include <map>

using namespace std;

typedef struct 
{
  double x;
  double y;
  double z;

}  coord3D_type;

typedef map<int, coord3D_type> VertexPos_type;


int main(int argc, char *argv[])
{
  
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " file.crd file.g \n";
        exit (1);
    }

    const char *coord3D_filename = argv[1];
    const char *edges_filename = argv[2];

    ifstream coord3D_file;
    coord3D_file.open( coord3D_filename );
    if (!coord3D_file)
    {
        cerr << "Error: [" << coord3D_filename << "] could not be opened.\n";
        exit(1);
    }
    ifstream edges_file;
    edges_file.open (edges_filename);
    if (!edges_file)
    {
        cerr << "Error: [" << edges_filename << "] could not be opened.\n";
        exit(1);
    }

    VertexPos_type V;

    // first read in vertex locations and  keep in map
    while (!coord3D_file.eof())
    {
        int node_num = 0;
        coord3D_type pos;

        coord3D_file >> node_num >> pos.x >> pos.y >> pos.z;
        V[node_num] = pos;
    }
    coord3D_file.close();

    // now write out the file
    
    // first, list all of the nodes as "points" in SAVG format

    cout << "points\n" ;
    for (VertexPos_type::const_iterator v = V.begin(); v!=V.end(); v++)
    {
         const coord3D_type &p = v->second;
         cout << p.x << " " << p.y << " " << p.z << "\n";
    }
    cout << "\n";

    // now list the edges
    //
    while ( !edges_file.eof())
    {
       int to = 0;
       int from = 0;

        edges_file >> from  >> to;

        const coord3D_type &from_3D = V[from];
        const coord3D_type &to_3D = V[to];

        cout << "lines\n" ;
        cout << from_3D.x << " " << from_3D.y << " " << from_3D.z << "\n";
        cout << to_3D.x << " " << to_3D.y << " " << to_3D.z << "\n";
        cout << "\n";
    }
    edges_file.close();

    return 0;
}
