#ifndef REACTION_HPP
#define REACTION_HPP
#include <vector>

#include "reactions_list.hpp"
#include "specie.hpp"
#include "context.hpp"

using namespace std;


enum reaction_id {
#define REACTION(name) name, 
LIST_OF_REACTIONS
#undef REACTION
  NUM_REACTIONS  //And a terminal marker so that we know how many there are
};

typedef std::pair<int, int> ReactionTerm;

template<class RATETYPE, class IMPL>
class reaction_base{
 public:
  RATETYPE active_rate(const Context<double>& c) const;
  RATETYPE rate;
  RATETYPE delay;
    
};

template<reaction_id RID>
class reaction : public reaction_base<double, reaction<RID> > {
 public:
  reaction();
  double active_rate(const Context<double>& c) const;
 protected:
  int num_inputs, num_outputs;
  const int* in_counts;
  const specie_id* inputs;
  const int* out_counts;
  const specie_id* outputs;
};


//And by the way, all of these will be declared at some point

#define REACTION(name) \
extern const int num_inputs_##name; \
extern const int num_outputs_##name; \
extern const int in_counts_##name[]; \
extern const specie_id inputs_##name[]; \
extern const int out_counts_##name[]; \
extern const specie_id outputs_##name[];

LIST_OF_REACTIONS
#undef REACTION

#endif

