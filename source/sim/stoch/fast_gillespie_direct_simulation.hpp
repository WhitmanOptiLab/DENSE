#ifndef SIM_STOCH_FAST_HPP
#define SIM_STOCH_FAST_HPP

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
class Fast_Gillespie_Direct_Simulation : public Simulation {

public:

    using Context = dense::Context<Fast_Gillespie_Direct_Simulation>;

 private:

    //"event" represents a delayed reaction scheduled to fire later
    struct event {
      Minutes time;
	     Natural cell;
      reaction_id rxn;
	    friend bool operator<(event const& a, event const &b) { return a.time < b.time;}
	    friend bool operator>(event const& a, event const& b) { return b < a; }
    };

    //"event_schedule" is a set ordered by time of delay reactions that will fire
    std::priority_queue<event, std::vector<event>, std::greater<event>> event_schedule;
    //"concs" stores current concentration levels for every species in every cell
    std::vector<std::vector<int> > concs;
    //"propensities" stores probability of each rxn firing, calculated from active rates
    std::vector<std::vector<Real>> propensities;
    //for each rxn, stores intracellular reactions whose rates are affected by a firing of that rxn
    std::vector<reaction_id> propensity_network[NUM_REACTIONS];
    //for each rxn, stores intercellular reactions whose rates are affected by a firing of that rxn
    std::vector<reaction_id> neighbor_propensity_network[NUM_REACTIONS];
    //random number generator
    std::default_random_engine generator = std::default_random_engine{ std::random_device()() };

    Real total_propensity_ = {};
    static std::uniform_real_distribution<Real> distribution_;

    Minutes generateTau();
    Minutes getSoonestDelay() const;
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
    Fast_Gillespie_Direct_Simulation(const Parameter_Set& ps, Real* pnFactorsPert, Real** pnFactorsGrad, int seed, std::vector<int> conc, NGraph::Graph adj_graph, dense::Natural num_grow_cell = 0)
    : Simulation(ps, std::move(adj_graph), pnFactorsPert, pnFactorsGrad, num_grow_cell)
    , concs(cell_count(), conc)
    , propensities(cell_count())
    , generator{seed} {
      initPropensityNetwork();
      initPropensities();
    }

    Real get_concentration (dense::Natural cell, specie_id species) const {
      try { 
        concs.at(cell).at(species);
      } catch (exception& error) {
        throw(std::out_of_range("Tried to fetch concentration of physical cell " + std::to_string(cell) + ", which is out of bounds"));
      }
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
    
    //add_cell: takes two cells in virtual id form and makes new cell from the parent cells history
    void add_cell(Natural cell, Natural parent = 0){
      Natural cell_index = add_cell_base(cell);; //new_index is the physical id for the virtual cell
      Natural parent_index = find_id(parent); //parent_index is the physical id for the parent virtual cell
      if(cell_index >= Natural(concs.size())){
        concs.push_back(concs[parent_index]);
      } else {
        concs[cell_index] = concs[parent_index];
      }
      cell_parameters_[cell_index] = cell_parameters_[parent_index];
    }
  
    //remove_cell: takes a virtual id and removes it from the simulation, setting it's propensities and concentrations to zero
    void remove_cell(Natural cell){
      Natural cell_index = remove_cell_base(cell);
      for(auto& spe : concs[cell_index]){
        spe = 0;
      }
      for(auto& rxn : propensities[cell_index]){
        rxn = 0;
      }
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
    CUDA_AGNOSTIC
    __attribute_noinline__ int choose_reaction(Real propensity_portion) {
      Real sum = 0;
      Natural c = 0;
      for (Natural virtual_cell : physical_cells_id()) {
        if(virtual_cell >= 0){
          for (Natural s = {}; s < NUM_REACTIONS; ++s) {
            sum += propensities[c][s];
            if (sum > propensity_portion) {
              return (c * NUM_REACTIONS) + s;
            }
          }
        }
        c++;
      }
      return cell_count() * NUM_REACTIONS - 1;
    }

    /*
     * UPDATEPROPENSITIES
     * recalculates the propensities of reactions affected by the firing of "rid"
     * arg "rid": the reaction that fired
    */
    CUDA_AGNOSTIC
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
                for (dense::Natural n=0; n < neighbor_count_by_cell_[cell_]; n++) { \
                    int n_cell = neighbors_by_cell_[cell_][n]; \
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
          for (Natural n = 0; n < neighbor_count_by_cell_[cell_]; ++n) {
            Natural n_cell = neighbors_by_cell_[cell_][n];
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
    for (dense::Natural i = 0; i < neighbor_count_by_cell_[cell]; ++i) {
      sum += concs[neighbors_by_cell_[cell][i]][species];
    }
    Real avg = sum / neighbor_count_by_cell_[cell];
    return std::isnan(avg) ? 0 : avg;
  }

  Minutes age_by(Minutes duration);

  private:

    Minutes time_until_next_event () const;

};

}

#endif
