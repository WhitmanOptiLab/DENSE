/**

    This is like g2Pmetric, but prints out the full distribution in 
    matrix format (coordinate storage).  For example, if at degree 5,
    we have three cluster freqeuncies  (1, 245)   (2, 7)  (3, 10), it
    would be represented as

    5 1  245
    5 2    7
    5 3   10

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
#include "tnt_stopwatch.h"

using namespace std;
using namespace NGraph;

typedef unsigned int degree;
typedef unsigned int freq_count;
typedef unsigned int cluster_size;
typedef double clustering_coeff;
typedef double conductance_t ;

int main(int argc, char *argv[])
{
    bool print_matrix = true;
    bool compute_avg_sparsity = false;
    bool compute_avg_conductance = false;
    bool print_timing = false;
    TNT::Stopwatch Q_total, Q_read, Q_print, Q_compute;;

    if (argc > 1)
    {
        string arg1(argv[1]);
        if (arg1 == "-c")
          compute_avg_conductance = true;
        else if (arg1 == "-s")
          compute_avg_sparsity = true;
        else if (arg1 == "-t")
          print_timing = true; 
    }
    
    Q_total.start();

    Graph G;
    
    Q_read.start(); 
    std::cin >> G;
    Q_read.stop();


    Q_compute.start();
    // create a frequency distribution of nodes based on degree
    typedef map< unsigned int, set<Graph::vertex> > DegreeList;
    DegreeList D;
    
    for ( Graph::const_iterator p = G.begin(); p!=G.end(); p++)
    {
      D[Graph::in_degree(p) + Graph::out_degree(p)].insert(Graph::node(p));
    }


    equivalence<Graph::vertex> E;
    set<Graph::vertex> V;    // this is the set of nodes in all components 
                             // so far (E)
    string input_line;

    E.recording_on();

    for (DegreeList::const_iterator p=D.begin(); p!=D.end(); p++)
    {
    
        Q_compute.resume();
        degree d = p->first;
        const set<Graph::vertex> &L = p->second;

        // now find weak component set of this subgraph

        weak_components_increment(G, E, V, L.begin(), L.end());


        if (print_matrix)
        {
            Q_compute.resume();
            cout << std::setprecision(2);
            map<cluster_size, freq_count> F;
            map<cluster_size, clustering_coeff> S; // average sparsity
            map<cluster_size, conductance_t> C;    // average conductance

            // create a frequency histrogram of cluster sizes
            for (equivalence<Graph::vertex>::const_iterator p=E.begin();
                    p != E.end(); p++)
            {
                  F[ p->second.size() ]++;
                  const Graph::vertex_set &v = p->second;

                  if (compute_avg_sparsity)
                  {
                     S[ v.size() ] +=  G.subgraph_sparsity( v );
                  } 
                  if (compute_avg_conductance)
                  {
                     C[ v.size()] += conductance(G, v);
                  }
                 
            }

            Q_compute.stop();
         
            Q_print.resume();
            //cout << ": [" << F.size() << "] : "  ;
            for (map<unsigned int, unsigned int>::const_iterator f=F.begin();
                    f!=F.end(); f++)
            {
                cluster_size z = f->first;
                freq_count freq = f->second;
                cout << d <<  " " << z << " " << freq;
                if (compute_avg_sparsity)
                {
                    cout << " " << S[z] / freq;
                }
                if (compute_avg_conductance)
                {
                    cout << " " << C[z] / freq;
                }
                 cout << "\n";
            }
            Q_print.stop();
        }

    }

    if (E.is_recording())
    {
          //
    }
     
    Q_total.stop();
    Q_print.stop();
    Q_compute.stop();
    cerr << "read time    : " << Q_read.read() << " secs\n";
    cerr << "compute time : " << Q_compute.read() << " secs\n";
    cerr << "print time   : " << Q_print.read() << " secs\n";
    cerr << "Total time   : " << Q_total.read() << " secs\n";

    return 0;
}
