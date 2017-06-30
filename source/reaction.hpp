#ifndef REACTION_HPP
#define REACTION_HPP

#include "specie.hpp"
#include "common_utils.hpp"
#include <utility>

//#include "simulation.hpp"
//#include "context.hpp"

using namespace std;


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


class reaction_base{
 public:
  CPUGPU_FUNC
  reaction_base(int inputs_num, int outputs_num, int factors_num, const int* inCount,
                const int* outCount, const specie_id* input_species, const specie_id* output_species,
                const specie_id* factor_species) :
                num_inputs(inputs_num), num_outputs(outputs_num),num_factors(factors_num),in_counts(inCount),
                out_counts(outCount), inputs(input_species), outputs(output_species), factors(factor_species){}
  CPUGPU_FUNC
  int getNumInputs() const { return num_inputs; }
  CPUGPU_FUNC
  int getNumFactors() const { return num_factors; }
  CPUGPU_FUNC
  int getNumOutputs() const { return num_outputs; }
  CPUGPU_FUNC
  const specie_id* getInputs() const { return inputs; }
  CPUGPU_FUNC
  const specie_id* getFactors() const { return factors; }
  CPUGPU_FUNC
  const specie_id* getOutputs() const { return outputs; }
  CPUGPU_FUNC
  const int* getInputCounts() const { return in_counts; }
  CPUGPU_FUNC
  const int* getOutputCounts() const { return out_counts; }

 protected:
  int num_inputs, num_outputs, num_factors;
  const int* in_counts;
  const specie_id* inputs;
  const int* out_counts;
  const specie_id* outputs;
  const specie_id* factors;
};

template<reaction_id RID>
class reaction : public reaction_base {
 public:
  reaction();
  template<class Ctxt>
  CPUGPU_FUNC
  RATETYPE active_rate(const Ctxt& c) const;
};

/*
static delay_reaction_id get_delay_reaction_id(reaction_id rid) {
  switch (rid) {
#define REACTION(name)
#define DELAY_REACTION(name) \
    case name : return dreact_##name; 

#include "reactions_list.hpp"
#undef REACTION
#undef DELAY_REACTION
    default: return NUM_DELAY_REACTIONS;
  }
};

*/

//And by the way, all of these will be declared at some point

#define REACTION(name) \
extern STATIC_VAR int num_inputs_##name; \
extern STATIC_VAR int num_outputs_##name; \
extern STATIC_VAR int num_factors_##name; \
extern STATIC_VAR int in_counts_##name[]; \
extern STATIC_VAR specie_id inputs_##name[]; \
extern STATIC_VAR int out_counts_##name[]; \
extern STATIC_VAR specie_id outputs_##name[]; \
extern STATIC_VAR specie_id factors_##name[];

#include "reactions_list.hpp"
#undef REACTION

#endif

