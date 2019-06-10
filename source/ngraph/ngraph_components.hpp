#ifndef NGRAPH_COMPONENT_H
#define NGRAPH_COMPONENT_H

#include "ngraph.hpp"
#include "equivalence.hpp"


using namespace NGraph;

template <typename T>
equivalence<T> components(const  tGraph<T> &G)
{
    equivalence<T> E;

   for (typename tGraph<T>::const_iterator pv = G.begin();  pv != G.end(); pv++)
   {
      const typename tGraph<T>::vertex &v = tGraph<T>::node(pv);
      const typename tGraph<T>::vertex_set &in = tGraph<T>:: in_neighbors(pv);
      const typename tGraph<T>::vertex_set &out = tGraph<T>::out_neighbors(pv);
        
      if (in.size() == 0 && out.size() == 0)
      {     
          E.insert(v);
      }     
      else              
      {     
        if (in.size() > 0)
        {   
          E.insert(v, *in.begin());
          E.insert( in.begin(), in.end());
        }
        if (out.size() > 0)
        { 
          E.insert(v, *out.begin());
          E.insert( out.begin(), out.end());
        }
      }
   }

   return E;
}

#endif
// NGRAPH_COMPONENT_H
