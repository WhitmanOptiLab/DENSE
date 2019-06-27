#include <set>
#include <stack>
#include <algorithm>
#include "ngraph.hpp"
#include "set_ops.hpp"

using namespace NGraph;

typedef unsigned int UInt;
typedef Graph::vertex Ge;
typedef Graph::vertex_neighbor_const_iterator vn_iter;

class SCC
{
    private:

      const Graph &G_;
      std::stack<Ge> S_;
      std::set<Ge> SS_;       // set version of S to make look-up faster
      std::vector<Ge> LowLink_;
      std::vector<Ge> Index_;
      UInt index_;

      UInt SCC_count_;
      std::vector<UInt> SCC_index_;

      void tarjan_(Ge v)
      {
        Index_[v] = index_;
        LowLink_[v] = index_;
        index_++;
        S_.push(v);
        SS_.insert(v);
        for (vn_iter p = G_.out_neighbors_begin(v); 
                p!=G_.out_neighbors_end(v); p++)
        {
          Graph::vertex vprime = *p;
          if (Index_[vprime] == 0)
          {
            tarjan_(vprime);
            LowLink_[v] = std::min( LowLink_[v], LowLink_[vprime]);
          }
          else if (includes_elm(SS_, vprime))
          {
            LowLink_[v] = std::min( LowLink_[v], Index_[vprime]);
          }
        }
        if (LowLink_[v] == Index_[v])
        {
            Ge vp = 0;
            SCC_count_++;
            do
            {
                vp = S_.top();  S_.pop();
                SS_.erase(vp);
                std::cout << vp << " " ;
                SCC_index_[vp] = SCC_count_;
            }
            while (vp != v );
            std::cout << "\n";
        }
      }

  public:

    SCC(const Graph &G): G_(G), S_(), SS_(), LowLink_(G.num_vertices()), 
            Index_(G.num_vertices()), index_(1), SCC_count_(0), 
            SCC_index_(G.num_vertices())
    {

        for (Graph::const_iterator v = G.begin(); v != G.end(); v++)
        { 
          if (Index_[Graph::node(v)] == 0)
             tarjan_(Graph::node(v));
        }
    }


    UInt component(Ge v)  const
    {
      return SCC_index_[v];
    }

};




