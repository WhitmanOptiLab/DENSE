#ifndef CLUSTER_COEFF_H
#define CLUSTER_COEFF_H

#include "ngraph.hpp"
#include "set_ops.hpp"



//
// cluster coefficient at a vertex
//

namespace NGraph{

template <typename T>
double cluster_coeff(typename tGraph<T>::const_iterator t)
{
	const typename tGraph<T>::vertex_set &neighbors = 
                    tGraph<T>::out_neighbors(t);
	int n = neighbors.size();

	if (n < 2) return 0.0;

	unsigned int num_edges = 0;
	//for each vertex in neighbors of a
	//
	for ( typename tGraph<T>::vertex_set::const_iterator p=neighbors.begin() ; 
					p != neighbors.end(); p++ )
	{
		num_edges += intersection_size(tGraph<T>::out_neighbors(p),neighbors);
	}
  return  static_cast<double>(num_edges) / (n*(n-1));
}


template <typename T>
double cluster_coeff(const tGraph<T> &A,  typename tGraph<T>::vertex &a)
{
  typename tGraph<T>::const_iterator p = A.find(a);

  return cluster_coeff(p);
}



template <typename T>
double cluster_coeff_directed(typename tGraph<T>::const_iterator t)
{

	const  typename tGraph<T>::vertex_set &neighbors = 
            tGraph<T>::in_neighbors(t) +  tGraph<T>::out_neighbors(t);

	int n = neighbors.size();

	if (n < 2) return 0.0;

	int num_edges = 0;
	//for each vertex in neighbors of a
	//
	for ( typename tGraph<T>::vertex_set::const_iterator p=neighbors.begin() ; 
					p != neighbors.end(); p++ )
	{
		num_edges += intersection_size(tGraph<T>::out_neighbors(p),neighbors); 
	}
  return  static_cast<double>(num_edges) / (n*(n-1));
}



template <typename T>
double cluster_coeff_directed(const tGraph<T> &A, 
                  typename tGraph<T>::vertex &a)
{
  typename tGraph<T>::const_iterator p = A.find(a);

  return cluster_coeff_directed(p);
}


//
// cluster coefficient of a graph (is the average of each node)
//
template <typename T>
double cluster_coeff_undirected(const tGraph<T> &A)
{

  double sum = 0.0;
  for ( typename tGraph<T>::const_iterator p = A.begin(); p != A.end(); p++)
  {
      sum += cluster_coeff_undirected(p); 
  }
  return  sum / A.num_vertices();
}


}
// namespace NGraph

#endif
// CLUSTER_COEFF_H
