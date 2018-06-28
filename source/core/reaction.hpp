#ifndef CORE_REACTION_HPP
#define CORE_REACTION_HPP

#include "specie.hpp"
#include "utility/cuda.hpp"
#include "utility/numerics.hpp"

#include <string>
#include <utility>


enum Reaction_ID {
  #define REACTION(name) name,
  #include "reactions_list.hpp"
  #undef REACTION
  Reaction_ID_size
};

constexpr auto NUM_REACTIONS = Reaction_ID_size;

using reaction_id = Reaction_ID;

const std::string reaction_str[NUM_REACTIONS] = {
    #define REACTION(name) #name,
    #include "reactions_list.hpp"
    #undef REACTION
};

enum delay_reaction_id {
#define REACTION(name)
#define DELAY_REACTION(name) dreact_##name,
#include "reactions_list.hpp"
#undef REACTION
#undef DELAY_REACTION
  NUM_DELAY_REACTIONS
};

using ReactionTerm = std::pair<int, int>;

class reaction_base{
 public:
  IF_CUDA(__host__ __device__)
  reaction_base(int specie_delta_num, int const* coeffs, specie_id const* ids) :
                num_deltas(specie_delta_num), deltas(coeffs), delta_ids(ids) {}
  IF_CUDA(__host__ __device__)
  int getNumDeltas() const { return num_deltas; }
  IF_CUDA(__host__ __device__)
  specie_id const* getSpecieDeltas() const { return delta_ids; }
  IF_CUDA(__host__ __device__)
  int const* getDeltas() const { return deltas; }

 private:
  int num_deltas;
  int const* deltas;
  specie_id const* delta_ids;
};

template<reaction_id RID>
class reaction : public reaction_base {
 public:
  reaction();
  template<class Ctxt>
  IF_CUDA(__host__ __device__)
  static Real active_rate(Ctxt const& c);
};

#define REACTION(name) template<> reaction<name>::reaction();
#include "reactions_list.hpp"
#undef REACTION


//And by the way, all of these will be declared at some point

#define REACTION(name) \
extern STATIC_VAR int num_deltas_##name; \
extern STATIC_VAR int deltas_##name[]; \
extern STATIC_VAR specie_id delta_ids_##name[];
#include "reactions_list.hpp"
#undef REACTION
#endif
