#ifndef REACTION_HPP
#define REACTION_HPP
#include <vector>

#include "reactions_list.h"

using namespace std;


enum reaction_id {
#define REACTION(name) name, 
LIST_OF_REACTIONS
#undef REACTION
  NUM_REACTIONS  //And a terminal marker so that we know how many there are
};

class Context;

typedef std::pair<int, int> ReactionTerm;

template<class RATETYPE, class IMPL>
class reaction_base{
 public:
  RATETYPE active_rate(const Context& c);
    
 protected:
  RATETYPE rate;
  RATETYPE delay;
  vector<ReactionTerm> inputs;
  vector<ReactionTerm> outputs;
    
};

template<reaction_id RID>
class reaction : public reaction_base<double, reaction<RID> > {
 public:
  reaction(double rate_in, double delay_in, vector<ReactionTerm> inputs_in, vector<ReactionTerm> outputs_in);
  double active_rate(const Context& c);
};

#endif

