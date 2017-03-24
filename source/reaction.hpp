#ifndef REACTION_HPP
#define REACTION_HPP
#include <vector>

#include "specie.hpp"
//#include "context.hpp"

using namespace std;
typedef float RATETYPE;
class Context;

enum reaction_id {
#define REACTION(name) name, 
#include "reactions_list.hpp"
#undef REACTION
  NUM_REACTIONS  //And a terminal marker so that we know how many there are
};

enum delay_reaction_id {
#define REACTION(name)
#define DELAY_REACTION(name) dreact_##name,
#include "reactions_list.hpp"
#undef REACTION
#undef DELAY_REACTION
  NUM_DELAY_REACTIONS
};

typedef std::pair<int, int> ReactionTerm;


template<class IMPL>
class reaction_base{
 public:
  RATETYPE active_rate(const Context& c) const;
  RATETYPE rate;
  RATETYPE delay;
    
};

template<reaction_id RID>
class reaction : public reaction_base<reaction<RID> > {
 public:
  reaction();
  RATETYPE active_rate(const Context& c) const;
  int getNumInputs() const { return num_inputs; }
  int getNumFactors() const { return num_factors; }
  int getNumOutputs() const { return num_outputs; }
  const specie_id* getInputs() const { return inputs; }
  const specie_id* getFactors() const { return factors; }
  const specie_id* getOutputs() const { return outputs; }
  const int* getInputCounts() const { return in_counts; }
  const int* getOutputCounts() const { return out_counts; }

 protected:
  int num_inputs, num_outputs, num_factors;
  const int* in_counts;
  const specie_id* inputs;
  const int* out_counts;
  const specie_id* outputs;
  const specie_id* factors;
};


//And by the way, all of these will be declared at some point

#define REACTION(name) \
extern const int num_inputs_##name; \
extern const int num_outputs_##name; \
extern const int num_factors_##name; \
extern const int in_counts_##name[]; \
extern const specie_id inputs_##name[]; \
extern const int out_counts_##name[]; \
extern const specie_id outputs_##name[]; \
extern const specie_id factors_##name[];

#include "reactions_list.hpp"
#undef REACTION

#endif

