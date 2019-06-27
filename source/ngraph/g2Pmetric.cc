/**

    Create the weakly-connected components of an input graph, given a specific
    order of nodes (one, or several per line) and report number of components
    at each step.

    Usage:

      cat foo.g | g2Pmetric  min-cluster-size   max-cluster-size


    Output:  the number of weakly-connected components (each number 
    coresponds to the original node list, e.g.

    [degree] [#clusters] [#total_cluster_sizes] [#max_cluster_size] [#vertices]


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



using namespace std;
using namespace NGraph;

int main(int argc, char *argv[])
{

    bool print_distribution = false;
    bool print_clustering = false;

    if (argc <= 1)
      exit(1);

    const unsigned int min_cluster_size = atoi(argv[1]);
    const unsigned int max_cluster_size = atoi(argv[2]);

    if (argc > 3)
    {
      string arg(argv[3]);
      if (arg == "-d")
      {
        print_distribution = true;
      }
      else if (arg == "-c")
      {
        print_distribution = true;
        print_clustering = true;
      }
    }

    Graph G;
    std::cin >> G;

    // create a frequency distribution of nodes based on degree
    typedef map< unsigned int, set<Graph::vertex> > DegreeList;
    DegreeList D;


    for ( Graph::const_iterator p = G.begin(); p!=G.end(); p++)
    {
      D[Graph::in_degree(p) + Graph::out_degree(p)].insert(Graph::node(p));
    }


    equivalence<Graph::vertex> Q;
    set<Graph::vertex> V;
    string input_line;

    for (DegreeList::const_iterator p=D.begin(); p!=D.end(); p++)
    {
     
        unsigned int degree = p->first;
        const set<Graph::vertex> &L = p->second;
        V.insert(L.begin(), L.end());

        // now find weak component set of this subgraph

        weak_components_increment(G, Q, V, L.begin(), L.end());


        vector<unsigned int> V = Q.class_sizes();

        unsigned int Max_size = *max_element(V.begin(), V.end());

        unsigned int num_nontrivial_clusters = 0;    
        unsigned int size_nontrivial_clusters = 0;    

        for (equivalence<Graph::vertex>::const_iterator p=Q.begin();
                    p != Q.end(); p++)
        {
          unsigned cluster_size = p->second.size();
          if ( (cluster_size >= min_cluster_size) && 
                    (cluster_size <= max_cluster_size))
          {
             num_nontrivial_clusters++;
             size_nontrivial_clusters += p->second.size();

          }
        }
        
        cout << degree << " " 
          << num_nontrivial_clusters << " " 
          << size_nontrivial_clusters << " " 
          << Max_size << " " << Q.num_elements();

        // now print out distribution (if -d flag was set)
        if (print_distribution)
        {
            cout << std::setprecision(2);
            map<unsigned int, unsigned int> F;
            map<unsigned int, double> S;      // average sparsity
            // create a frequency histrogram of cluster sizes
            for (equivalence<Graph::vertex>::const_iterator p=Q.begin();
                    p != Q.end(); p++)
            {
                  F[ p->second.size() ]++;

                  if (print_clustering)
                    S[ p->second.size()] += G.subgraph_sparsity(p->second);
            }

         
            cout << ": [" << F.size() << "] : "  ;
            for (map<unsigned int, unsigned int>::const_iterator f=F.begin();
                    f!=F.end(); f++)
            {
                cout << "(" << f->first << ", " << f->second ;
                if (print_clustering)                
                   cout << ", " << S[f->first]/(f->second) ;
                cout << ") ";
            }
        }

        cout << "\n";
    }

    return 0;
      
}
