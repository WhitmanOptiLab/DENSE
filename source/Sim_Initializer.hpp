#ifndef SIM_INITIALIZER_HPP
#define SIM_INITIALIZER_HPP

#include "io/arg_parse.hpp"
#include "measurement/oscillation.hpp"
#include "measurement/basic.hpp"
#include "measurement/bad_simulation_error.hpp"
#include "utility/style.hpp"
#include "utility/common_utils.hpp"
#include "io/csvr_sim.hpp"
#include "io/csvw_sim.hpp"
#include "model_impl.hpp"
#include "io/ezxml/ezxml.h"
#include "ngraph/ngraph_components.hpp"

using style::Color;

#include <chrono>
#include <cstdlib>
#include <cassert>
#include <random>
#include <memory>
#include <iterator>
#include <algorithm>
#include <functional>
#include <exception>
#include <iostream>
#include <cmath>

namespace dense{

struct Static_Args_Base {
  Real*  perturbation_factors; 
  Real**  gradient_factors;
  std::chrono::duration<Real, std::chrono::minutes::period> simulation_duration;
  std::chrono::duration<Real, std::chrono::minutes::period> analysis_interval;
  int help;
  NGraph::Graph adj_graph;
};

struct Static_Args : public Static_Args_Base {
  std::vector<Parameter_Set> param_sets;
};

struct Param_Static_Args : public Static_Args_Base {
	Parameter_Set lbounds;
	Parameter_Set ubounds;
	int pop;
	int parent;
	std::vector<Real> real_input;
	int num_generations;
};
  
void parse_graphml(const char* file, NGraph::Graph* adj_graph);
void create_default_graph(NGraph::Graph* a_graph, int cell_total, int tissue_width);
void graph_constructor(Static_Args_Base* param_args, std::string string_file, int cell_total, int tissue_width);
  
template <typename NUM_TYPE>
void conc_vector(std::string init_conc, bool c_or_0, std::vector<NUM_TYPE>* conc){
  if(c_or_0){
    NUM_TYPE c_species;
    csvr reader = csvr(init_conc);
    while(reader.get_next(&c_species)){
      conc->push_back(c_species);
    }
    while(conc->size() > NUM_SPECIES){
      conc->pop_back();
    }
    while(conc->size() < NUM_SPECIES){
      conc->push_back(0);
    }
  } else {
    for(int i = 0; i < NUM_SPECIES; i++){
      conc->push_back(0);
    }
  }
}

}

#endif
