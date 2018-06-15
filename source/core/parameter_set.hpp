#ifndef CORE_PARAM_SET_HPP
#define CORE_PARAM_SET_HPP

#include "specie.hpp"
#include "reaction.hpp"
#include "io/csvr.hpp"

#include <string>


constexpr auto NUM_PARAMS = NUM_CRITICAL_SPECIES + NUM_REACTIONS + NUM_DELAY_REACTIONS;

class Parameter_Set {
 private:
    //From the old code - not in use
    //char* print_name; // The mutant's name for printing output
    //char* dir_name; // The mutant's name for making its directory
    //int num_knockouts; // The number of knockouts required to achieve this mutant
    //int knockouts[2]; // The indices of the concentrations to knockout (num_knockouts determines how many indices to knockout)
    //double overexpression_rate; // Which gene should be overexpressed, -1 = none
    //double overexpression_factor; // Overexpression factors with 1 = 100% overexpressed, if 0 then no overexpression
    //int induction; // The induction point for mutants that are time sensitive
    //int recovery;
    //int num_conditions[NUM_SECTIONS + 1]; // The number of conditions this mutant is tested on
    //double cond_scores[NUM_SECTIONS + 1][MAX_CONDS_ANY]; // The score this mutant can achieve for each condition
    //double max_cond_scores[NUM_SECTIONS]; // The maximum score this mutant can achieve for each section
    //bool secs_passed[NUM_SECTIONS]; // Whether or not this mutant has passed each section's conditions
    //double conds_passed[NUM_SECTIONS][1 + MAX_CONDS_ANY]; // The score this mutant achieved for each condition when run
    //feature feat; // The oscillation features this mutant produced when run
    //int print_con; // The index of the concentration that should be printed (usually mh1)
    //bool only_post; //indicating if only the posterior is simulated

    // currently in use
  RATETYPE _parameters[NUM_PARAMS + 1];

 public:
  Parameter_Set() {};

  RATETYPE getCriticalValue(critspecie_id i) const { return _parameters[NUM_REACTIONS + NUM_DELAY_REACTIONS + i]; }
  RATETYPE getDelay(delay_reaction_id i) const { return _parameters[NUM_REACTIONS + i]; }
  RATETYPE getReactionRate(reaction_id i) const { return _parameters[i]; }
  RATETYPE* getArray() { return _parameters; }
  RATETYPE const* getArray() const { return _parameters; }

  Real* begin() { return _parameters + 0; }
  Real* end() { return _parameters + NUM_PARAMS; }

  bool import_from (csvr & in) {
    for (auto & parameter : *this) {
      if (!in.get_next(&parameter)) return false;
    }
    return true;
  }

  void printall() const
  {
    for (unsigned int i = 0; i < NUM_CRITICAL_SPECIES; ++i)
      printf("%f, ", getReactionRate(reaction_id(i)));
    printf("\n");
    for (unsigned int i = 0; i < NUM_DELAY_REACTIONS; ++i)
      printf("%f, ", getDelay(delay_reaction_id(i)));
    printf("\n");
    for (unsigned int i = 0; i < NUM_REACTIONS; ++i)
      printf("%f, ", getCriticalValue(critspecie_id(i)));
    printf("\n");
  }
};

#endif
