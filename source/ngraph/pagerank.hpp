#ifndef  NGRAPH_PAGERANK_H
#define NGRAPH_PAGERANK_H

#include <vector>
#include <math.h>
#include "ngraph.hpp"



using namespace NGraph;
using namespace std;

template <typename Graph>
double pagerank_one_iteration(const Graph &G, 
      map<typename Graph::vertex, double> &P, double damping)
{

    typedef typename Graph::const_iterator const_iterator;
    typedef typename Graph::vertex_set  vertex_set;


    double max_delta = 0.0;
    for (const_iterator v=G.begin(); v!= G.end(); v++)
    {
       double sum = 0.0;
       double old_pagerank = P[Graph::node(v)];

        const vertex_set &E = Graph::in_neighbors(v);
        for (typename vertex_set::const_iterator e = E.begin(); e!=E.end(); e++)
        {
           sum += P[*e] / G.out_degree(*e);
        }
        double new_pagerank = (1-damping) + damping * sum;
        P[Graph::node(v)] = new_pagerank;

        double delta = fabs((old_pagerank - new_pagerank)/old_pagerank) ;
        if (delta > max_delta)
            max_delta = delta;
    }

    return max_delta;
}


template <typename Graph>
map<typename Graph::vertex, double> 
       pagerank(const Graph &G, 
            unsigned int &iterations_used,
            unsigned int max_iterations = 100, 
            double max_delta = 0.01,
            double damping = 0.85,
            bool verbose = false)
{

    typedef typename Graph::vertex T;
    typedef typename Graph::const_iterator const_iterator;


    map<T,double> P;
    
    // initialize all P[i] to  1
    for ( const_iterator i = G.begin(); i!=G.end(); i++)
    {
         P[Graph::node(i)] = 1;
    }

    unsigned int i=0;    
    for (; i<max_iterations; i++)
    {
       double delta = pagerank_one_iteration(G, P, damping);
       if (verbose)
          cerr << i << " " << delta << "\n";
       if (delta <= max_delta)
          break;
    }
    iterations_used = i;

#if 0
    // normalize to size of graph (nodes)
    const unsigned N = G.num_vertices();
    for (typename map<T,double>::iterator t = P.begin(); t!=P.end(); t++)
    {
      t->second /= N;
    }
#endif


    return P;

}

// vector P refers to the pagerank values of each node in G, i.e. P[0]
// is the pagerank of the first node, G.begin()


#endif
// NGRAPH_PAGERANK_H
