// In this header file, define your model!
// This includes functions to describe each reaction.
// Make sure that you've first completed reaction_list.h and specie_list.h
#ifndef MODEL_IMPL_H
#define MODEL_IMPL_H
#include "core/reaction.hpp"
#include "core/specie.hpp"
#include "core/model.hpp"
#include "sim/base.hpp"
#include <cstddef>

/*

Define all reaction rate functions in `model_impl.hpp`.
For example, if a reaction is enumerated `R_ONE`, it should be declared as a 
   function like this:
    
   RATETYPE reaction<R_ONE>::active_rate(const Ctxt& c) const { return 6.0; }
     
   Or, for a more interesting reaction rate, you might do something like:
     
   RATETYPE reaction<R_TWO>::active_rate(const Ctxt& c) const {
            return c.getRate(R_TWO) * c.getCon(SPECIE_ONE) * c.neighbors.calculateNeighborAvg(SPECIE_TWO);
   }

Refer to the Context API below for instructions on how to get delays
and critical values for more complex reaction rate functions.
    
   Contexts are iterators over the concentration levels of all species in all cells. Use them to get the conc values of specific species that factor into reaction rate equations.
   
   To get the concentration of a specie where `c` is the context object and `SPECIE` is the specie's enumeration:
   `c.getCon(SPECIE)`
   
   To get the delay time of a particular delay reaction that is enumerated as `R_ONE` and is properly identified as a delay reaction in `reactions_list.hpp` (see 2.0.1):
   `RATETYPE delay_time = c.getDelay(dreact_R_ONE);`
       
   To get the past concentration of `SPECIE` where `delay_time`, as specified in the previous example, is the delay time for `R_ONE`:
   `c.getCon(SPECIE, delay_time);`
       
   To get average concentration of SPECIE in that cell and its surrounding cells:
   `c.calculateNeighborAvg(SPECIE)`
       
   To get the past average concentration of SPECIE in that cell and its surround cells:
   `c.calculateNeighborAvg(SPECIE, delay_time)`


EXAMPLE of two reactions:

template<>
template<class Ctxt>
RATETYPE reaction<ph1_synthesis>::active_rate(const Ctxt& c) const {
    return c.getRate(ph1_synthesis) * c.getCon(mh1,c.getDelay(dreact_ph1_synthesis));
}

template<>
template<class Ctxt>
RATETYPE reaction<ph1_degradation>::active_rate(const Ctxt& c) const {
    return c.getRate(ph1_degradation) * c.getCon(ph1);
}

*/


#endif // MODEL_IMPL_H
