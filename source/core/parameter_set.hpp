#ifndef CORE_PARAM_SET_HPP
#define CORE_PARAM_SET_HPP

#include "specie.hpp"
#include "reaction.hpp"
#include "io/csvr.hpp"

#include <string>


constexpr auto NUM_PARAMS = NUM_CRITICAL_SPECIES + NUM_REACTIONS + NUM_DELAY_REACTIONS;

constexpr auto rate_constants_offset_ = 0;
constexpr auto delays_offset_ = rate_constants_offset_ + NUM_REACTIONS;
constexpr auto critical_values_offset_ = delays_offset_ + NUM_DELAY_REACTIONS;

class Parameter_Set {

  public:

    using iterator = Real*;
    using const_iterator = Real const*;

    Parameter_Set () noexcept = default;

    Real getCriticalValue (critspecie_id i) const noexcept {
      return (parameters_ + critical_values_offset_)[i];
    }

    Real getDelay (delay_reaction_id i) const noexcept {
      return (parameters_ + delays_offset_)[i];
    }

    Real getReactionRate (reaction_id i) const noexcept {
      return (parameters_ + rate_constants_offset_)[i];
    }

    Real* data () noexcept {
      return parameters_;
    }

    Real const* data () const noexcept {
      return parameters_;
    }

    iterator begin () noexcept {
      return parameters_ + 0;
    }

    const_iterator begin () const noexcept {
      return parameters_ + 0;
    }

    iterator end () noexcept {
      return parameters_ + NUM_PARAMS;
    }

    const_iterator end () const noexcept {
      return parameters_ + NUM_PARAMS;
    }

  bool import_from (csvr & in) {
    for (auto & parameter : *this) {
      if (!in.get_next(&parameter)) return false;
    }
    return true;
  }

  friend std::istream& operator>> (std::istream& in, Parameter_Set& parameter_set) {
    for (auto & parameter : parameter_set) {
      if (!csvr::get_real(in, &parameter)) {
        in.setstate(std::ios_base::failbit);
        break;
      }
    }
    return in;
  }

  private:

    Real parameters_[NUM_PARAMS] = {};

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

};

#endif
