#ifndef RXN_STRUCT
#define RXN_STRUCT
#include <cmath>
#include "rxn_struct.hpp"
#include "sim/cell_param.hpp"
#include "model_impl.hpp"
#include "core/model.hpp"
#include <limits>
#include <iostream>
#include <cmath>
#include <set>

namespace dense{
namespace stochastic{
struct Rxn {
  dense::Natural cell;
  reaction_id reaction;
  Real upper_bound;
  Real lower_bound;
  friend bool operator==(const Rxn& a, const Rxn& b){ return (a.cell == b.cell && a.reaction == b.reaction);}
  friend bool operator<(const Rxn& a, const Rxn& b){return (a.upper_bound < b.upper_bound);}
  friend bool operator> (const Rxn& a, const Rxn& b){return (b < a);}
  friend bool operator<=(const Rxn& a, const Rxn& b){return((a < b) || (a.upper_bound == b.upper_bound)); }
  
  int get_group_rank() {if(upper_bound == 0) {return std::numeric_limits<int>::min();} else{return (int)(ceil(std::log2(upper_bound)));}}
  
};
  
struct Delay_Rxn {
  Rxn rxn;
  delay_reaction_id delay_reaction;
  Minutes delay;
  friend bool operator<(const Delay_Rxn& a, const Delay_Rxn b){return a.delay < b.delay;}
  friend bool operator>(const Delay_Rxn& a, const Delay_Rxn b){return b < a;}
};
}
}
#endif