#ifndef CORE_REACTION_HPP
#define CORE_REACTION_HPP

#include "specie.hpp"
#include "util/common_utils.hpp"
#include <string>
#include <utility>

using namespace std;


enum reaction_id {
#define REACTION(name) name, 
#include "reactions_list.hpp"
#undef REACTION
  NUM_REACTIONS  //And a terminal marker so that we know how many there are
};

const string reaction_str[NUM_REACTIONS] = {
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

typedef std::pair<int, int> ReactionTerm;


class reaction_base{
 public:
  CPUGPU_FUNC
  reaction_base(int specie_delta_num, const int* coeffs, const specie_id* ids) :
                num_deltas(specie_delta_num), deltas(coeffs), delta_ids(ids) {}
  CPUGPU_FUNC
  int getNumDeltas() const { return num_deltas; }
  CPUGPU_FUNC
  const specie_id* getSpecieDeltas() const { return delta_ids; }
  CPUGPU_FUNC
  const int* getDeltas() const { return deltas; }

 protected:
  int num_deltas;
  const int* deltas;
  const specie_id* delta_ids;
};

template<reaction_id RID>
class reaction : public reaction_base {
 public:
  reaction();
  template<class Ctxt>
  CPUGPU_FUNC
  RATETYPE active_rate(const Ctxt& c) const;
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

