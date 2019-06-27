/**

    This is like g2Pmetric, but prints out the full distribution in 
    matrix format (coordinate storage).  For example, if at degree 5,
    we have three cluster freqeuncies  (1, 245)   (2, 7)  (3, 10), it
    would be represented as

    3 1  245
    2 2    7
    3 3   10

    This can be read into matlab as a sparse matrix and plotted.


    Usage: 

       cat foo.g | g2Pmatrix



*/

#include <iostream>
#include <fstream>
#include <list>
#include <set>
#include <string>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include "ngraph.hpp"
#include "ngraph_weak.hpp"
#include "ngraph_conductance.hpp"


using namespace std;
using namespace NGraph;

typedef unsigned int degree;
typedef unsigned int freq_count;
typedef unsigned int cluster_size;
typedef double clustering_coeff;
typedef double conductance_t ;

int main(int argc, char *argv[])
{

    Graph G;
    std::cin >> G;

    // create a frequency distribution of nodes based on degree
    typedef map< unsigned int, set<Graph::vertex> > DegreeList;
    DegreeList D;


    for ( Graph::const_iterator p = G.begin(); p!=G.end(); p++)
    {
      D[Graph::in_degree(p) + Graph::out_degree(p)].insert(Graph::node(p));
    }


    typedef equivalence<Graph::vertex> Equiv;

    Equiv E;
    set<Graph::vertex> V;    // this is the set of nodes in all components 
                             // so far (E)
    string input_line;

    E.recording_on();

    for (DegreeList::const_iterator p=D.begin(); p!=D.end(); p++)
    {
     
        const set<Graph::vertex> &L = p->second;

        // now find weak component set of this subgraph

        weak_components_increment(G, E, V, L.begin(), L.end());

    }

    // now dump out merges
    typedef vector<Equiv::triplet> Merge_List;
    const Merge_List &M = E.merge_list();
    for (Merge_List::const_iterator p=M.begin(); p!=M.end(); p++)
    {
      cout << "(" << p->left << ", " << p->right << ")  -> " << p->to
                << "\n";
    }

    
    
    cout << "\n\n";

    // now dump out list of original equilance indices
    //
    typedef map<Graph::vertex, Equiv::index_t> E_list;
    const E_list &E1 = E.original_class_indices();
    for (E_list::const_iterator v = E1.begin(); v!=E1.end(); v++)
    {
      cout <<  v-> first << " : " << v->second << "\n";
    }
    

    return 0;  
}
