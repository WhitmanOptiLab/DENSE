#ifndef SIM_STOCH_HPP
#define SIM_STOCH_HPP

#include "sim/base.hpp"
#include "core/parameter_set.hpp"
#include "core/model.hpp"
#include "core/specie.hpp"
#include "sim/cell_param.hpp"
#include "core/reaction.hpp"
#include <vector>
#include <set>
#include <queue>
#include <random>

namespace dense {

/*
 * STOCHASTIC SIMULATOR:
 * superclasses: simulation_base, Observable
 * uses Gillespie's tau leaping algorithm
 * uses Barrio's delay SSA
*/
class Stochastic_Simulation : public Simulation {

public:

    using Context = dense::Context<Stochastic_Simulation>;

 private:

    //"event" represents a delayed reaction scheduled to fire later
    struct event{
      Real time;
	    int cell;
      reaction_id rxn;
	    Real getTime() const{return time;}
	    bool operator<(event const &b) const {return time < b.time;}
	    bool operator>(event const& other) const { return other < *this; }
    };

    //"event_schedule" is a set ordered by time of delay reactions that will fire
    std::priority_queue<event, std::vector<event>, std::greater<event>> event_schedule;

    //"concs" stores current concentration levels for every species in every cell
    std::vector<std::vector<int> > concs;
    //"propensities" stores probability of each rxn firing, calculated from active rates
    std::vector<std::vector<Real> > propensities;
    //for each rxn, stores intracellular reactions whose rates are affected by a firing of that rxn
    std::vector<reaction_id> propensity_network[NUM_REACTIONS];
    //for each rxn, stores intercellular reactions whose rates are affected by a firing of that rxn
    std::vector<reaction_id> neighbor_propensity_network[NUM_REACTIONS];
    //random number generator
    std::default_random_engine generator;

    Real total_propensity_ = 0.0;

    Real generateTau();
    Real getSoonestDelay();
    void executeDelayRXN();
    Real getRandVariable();
    void tauLeap();
    void initPropensityNetwork();
    void generateRXNTaus(Real tau);
    void fireOrSchedule(int c, reaction_id rid);
    void initPropensities();

    public:

    /*
     * ContextStoch:
     * iterator for observers to access conc levels with
    */
    using SpecieRates = CUDA_Array<Real, NUM_SPECIES>;

  private:
    void fireReaction(dense::Natural cell, const reaction_id rid);

  public:
    /*
     * Constructor:
     * calls simulation base constructor
     * initializes fields "t" and "generator"
    */
    Stochastic_Simulation(const Parameter_Set& ps, Real* pnFactorsPert, Real** pnFactorsGrad, int cells_total, int width_total, int seed)
    : Simulation(ps, cells_total, width_total, pnFactorsPert, pnFactorsGrad)
    , concs(_cells_total, std::vector<int>(NUM_SPECIES, 0))
    , propensities(_cells_total)
    , generator{std::default_random_engine(seed)} {
      initPropensityNetwork();
      initPropensities();
    }

    Real get_concentration (dense::Natural cell, specie_id species) const {
      return concs.at(cell).at(species);
    }

    Real get_concentration (dense::Natural cell, specie_id species, dense::Natural delay) const {
      (void)delay;
      return get_concentration(cell, species);
    }

    void update_concentration (dense::Natural cell_, specie_id sid, int delta) {
      auto& concentration = concs[cell_][sid];
      concentration = std::max(concentration + delta, 0);
    }


  /*
   * GETTOTALPROPENSITY
   * sums the propensities of every reaction in every cell
   * called by "generateTau" in simulation_stoch.cpp
   * return "sum": the propensity sum
  */
   // Todo: store this as a cached variable and change it as propensities change;
   // sum += new_value - old_value;
    __attribute_noinline__ Real get_total_propensity() const {
      Real sum = total_propensity_; // 0.0;
      /*for (dense::Natural c = 0; c < _cells_total; ++c) {
        for (int r=0; r<NUM_REACTIONS; r++) {
          sum += propensities[c][r];
        }
      }*/
      return sum;
    }


    /*
     * CHOOSEREACTION
     * randomly chooses a reaction biased by their propensities
     * arg "propensity_portion": the propensity sum times a random variable between 0.0 and 1.0
     * return "j": the index of the reaction chosen.
    */
    CUDA_HOST CUDA_DEVICE
    __attribute_noinline__ int choose_reaction(Real propensity_portion) {
      Real sum = 0;
      dense::Natural c;
      int s;

      for (c = 0; c < _cells_total; c++) {
        for (s = 0; s < NUM_REACTIONS; s++) {
          sum += propensities[c][s];

          if (sum > propensity_portion) {
            int j = (c * NUM_REACTIONS) + s;
            return j;
          }
        }
      }

      int j = ((c - 1) * NUM_REACTIONS) + (s - 1);
      return j;
    }

    /*
     * UPDATEPROPENSITIES
     * recalculates the propensities of reactions affected by the firing of "rid"
     * arg "rid": the reaction that fired
    */
    CUDA_HOST CUDA_DEVICE
    __attribute_noinline__ void update_propensities(dense::Natural cell_, reaction_id rid) {
        #define REACTION(name) \
        for (std::size_t i=0; i< propensity_network[rid].size(); i++) { \
            if ( name == propensity_network[rid][i] ) { \
                auto& p = propensities[cell_][name];\
                auto new_p = dense::model::reaction_##name.active_rate(Context(*this, cell_)); \
                total_propensity_ += new_p - p;\
                p = new_p;\
            } \
        } \
    \
        for (std::size_t r=0; r< neighbor_propensity_network[rid].size(); r++) { \
            if (name == neighbor_propensity_network[rid][r]) { \
                for (dense::Natural n=0; n< _numNeighbors[cell_]; n++) { \
                    int n_cell = _neighbors[cell_][n]; \
                    Context neighbor(*this, n_cell); \
                    auto& p = propensities[n_cell][name];\
                    auto new_p = dense::model::reaction_##name.active_rate(neighbor); \
                    total_propensity_ += new_p - p;\
                    p = new_p;\
                } \
            } \
        }
        #include "reactions_list.hpp"
        #undef REACTION
        /*for (auto rxn : propensity_network[rid]) {
          auto& p = propensities[cell_][rxn];
          auto new_p = dense::model::active_rate(rxn, Context(this, cell_));
          total_propensity_ += new_p - p;
          p = new_p;
        }
        for (auto rxn : neighbor_propensity_network[rid]) {
          for (Natural n = 0; n < _numNeighbors[cell_]; ++n) {
            Natural n_cell = _neighbors[cell_][n];
            auto& p = propensities[n_cell][rxn];
            auto new_p = dense::model::active_rate(rxn, Context(this, n_cell));
            total_propensity_ += new_p - p;
            p = new_p;
          }
        }*/
    }


  /*
   * CALCULATENEIGHBORAVG
   * arg "sp": the specie to average from the surrounding cells
   * arg "delay": unused, but used in deterministic context. Kept for polymorphism
   * returns "avg": average concentration of specie in current and neighboring cells
  */
  Real calculate_neighbor_average (dense::Natural cell, specie_id species, dense::Natural delay) const {
    (void)delay;
    Real sum = 0;
    for (dense::Natural i = 0; i < _numNeighbors[cell]; ++i) {
        sum += concs[_neighbors[cell][i]][species];
    }
    Real avg = sum / _numNeighbors[cell];
    return avg;
  }

    void simulate_for (Minutes duration) {
      return simulate_for(duration.count());
    }

    void simulate_for(Real duration);

};

}
#endif
