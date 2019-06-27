#ifndef CLUSTER_COEFF_H
#define CLUSTER_COEFF_H

#include "ngraph.hpp"
#include "set_ops.hpp"



//
// cluster coefficient at a vertex
//

namespace NGraph{

// cut size is the number of edges between cluster and rest of graph

template <typename T>
unsigned int cut_size(const tGraph<T> &G, 
        const typename  tGraph<T>::vertex_set &v)
{

  int res = 0;
  typedef typename tGraph<T>::vertex_set::const_iterator p_iterator;

  for (p_iterator p=v.begin(); p!=v.end(); p++)
  {
      typename tGraph<T>::const_iterator pG = G.find(*p);
      if (pG != G.end())
      {
        const set<T> &out = tGraph<T>::out_neighbors(pG);
        const set<T> &in  = tGraph<T>::in_neighbors(pG);

        res += set_difference_size( in, v);
        res += set_difference_size( out, v);
      }
  }
  return res;
}



// conductance is the ratio of cut-size normalized by the number internal-edges 
//
template <typename T>
double conductance(const tGraph<T> &G, 
        const typename tGraph<T>::vertex_set &v)
{
  return (v.size() < 3 ? 0.0 :
          static_cast<double>(cut_size(G,v)) / G.subgraph_size(v) );
}


// number of distinct neighbors outside the cluster
//
template <typename T>
unsigned int distinct_neighbors_size(const tGraph<T> &G, 
        const typename  tGraph<T>::vertex_set &v)
{

  set<T> distinct_neighbors;
  typedef typename tGraph<T>::vertex_set::const_iterator p_iterator;

  for (p_iterator p=v.begin(); p!=v.end(); p++)
  {
      typename tGraph<T>::const_iterator pG = G.find(*p);
      if (pG != G.end())
      {
        const set<T> &out = tGraph<T>::out_neighbors(pG);
        const set<T> &in  = tGraph<T>::in_neighbors(pG);

        if (out.size() > 0)
        {
          std::set_difference(out.begin(), out.end(), v.begin(), v.end(),
            inserter(distinct_neighbors, distinct_neighbors.begin()));
        }

        if (in.size() > 0)
        {
          std::set_difference(in.begin(), in.end(), v.begin(), v.end(),
            inserter(distinct_neighbors, distinct_neighbors.begin()));
        }
      }
  }
  return distinct_neighbors.size();
}


// alignment is the ratio of cut-size normalized by the number 
// internal-edges 
//
//  NOTE:  alignment < conductance
//
template <typename T>
double alignment(const tGraph<T> &G, 
        const typename tGraph<T>::vertex_set &v)
{
  double res = 0.0;
  if  (v.size() < 3)
      res = 0.0;
  else
  {
     double n = distinct_neighbors_size(G,v);
     double V = G.subgraph_size(v);
     // cerr << " Debug: " << n << "/" << V << "\n";

     res  = (V < 1 ? 0.0 : n / V);
  }
  return res;
}


}
// namespace NGraph

#endif
// CLUSTER_COEFF_H
