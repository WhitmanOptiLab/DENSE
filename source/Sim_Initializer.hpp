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
#include "sim/determ/determ.hpp"
#include "sim/stoch/fast_gillespie_direct_simulation.hpp"
#include "sim/stoch/next_reaction_simulation.hpp"
#include "model_impl.hpp"
#include "io/ezxml/ezxml.h"
#include "Sim_Builder.hpp"
#include "ngraph/ngraph_components.hpp"
#include "arg_parse.hpp"

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

using dense::csvw_sim;
using dense::CSV_Streamed_Simulation;
using dense::Deterministic_Simulation;
using dense::Fast_Gillespie_Direct_Simulation;
using dense::stochastic::Next_Reaction_Simulation;

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
  
void parse_graphml(const char* file, NGraph::Graph* adj_graph){
  ezxml_t graphml = ezxml_parse_file(file);
  ezxml_t _graph = ezxml_child(graphml, "graph");
  ezxml_t _edges = ezxml_child(_graph, "edges");
  ezxml_t _edge = ezxml_child(_edges, "edge");
  
  if(_edge == nullptr){
    _edge = ezxml_child(_graph, "edge");
  }
  
  while ( _edge ){
    const char* vertex1 = ezxml_attr(_edge, "vertex1");
    const char* vertex2 = ezxml_attr(_edge, "vertex2");
    const char* source = ezxml_attr(_edge, "source");
    const char* target = ezxml_attr(_edge, "target");
    
    std::string begin;
    std::string end;
    
    if(vertex1){
      begin = std::string(vertex1);
      end = std::string(vertex2);
    } else {
      begin = std::string(source);
      end = std::string(target);
    }
    
    if(begin[0] == 'n'){
      begin.erase(begin.begin());
      end.erase(end.begin());
    }
    
    adj_graph->insert_edge_noloop(stoi(begin), stoi(end));

    _edge = ezxml_next(_edge);
  }
}

void create_default_graph(NGraph::Graph* a_graph, int cell_total, int tissue_width){
  for (Natural i = 0; i < cell_total; ++i) {
      bool is_former_edge = i % tissue_width == 0;
      bool is_latter_edge = (i + 1) % tissue_width == 0;
      bool is_even = i % 2 == 0;
      Natural la = (is_former_edge || !is_even) ? tissue_width - 1 : -1;
      Natural ra = !(is_latter_edge || is_even) ? tissue_width + 1 :  1;

      Natural top          = (i - tissue_width      + cell_total) % cell_total;
      Natural bottom       = (i + tissue_width                   ) % cell_total;
      Natural bottom_right = (i                  + ra              ) % cell_total;
      Natural top_left     = (i                  + la              ) % cell_total;
      Natural top_right    = (i - tissue_width + ra + cell_total) % cell_total;
      Natural bottom_left  = (i - tissue_width + la + cell_total) % cell_total;

      if (is_former_edge) {
        a_graph->insert_edge(i,abs(top));
        a_graph->insert_edge(i,abs(top_left));
        a_graph->insert_edge(i,abs(top_right));
        a_graph->insert_edge(i,abs(bottom_left));
      } else if (is_latter_edge) {
        a_graph->insert_edge(i,abs(top));
        a_graph->insert_edge(i,abs(top_right));
        a_graph->insert_edge(i,abs(bottom_right));
        a_graph->insert_edge(i,abs(bottom));
      } else {
        a_graph->insert_edge(i,abs(top_right));
        a_graph->insert_edge(i,abs(bottom_right));
        a_graph->insert_edge(i,abs(bottom));
        a_graph->insert_edge(i,abs(top_left));
        a_graph->insert_edge(i,abs(bottom_left));
    }
  }
}
  
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

void graph_constructor(Static_Args_Base* param_args, std::string string_file, int cell_total, int tissue_width){
  NGraph::Graph a_graph;
  if(cell_total == 0 && tissue_width == 0){
    std::ifstream open_file(string_file);
    if( open_file ){
      if(string_file.find("graphml") != std::string::npos){
        parse_graphml(string_file.c_str(), &a_graph);
      }
      else{
        NGraph::Graph a = NGraph::Graph(open_file);
        a_graph = std::move(a);
      }
    } else {
      std::cout << style::apply(Color::red) << "Error: Could not find cell graph file " + string_file + " specified by the -f command. Make sure file is spelled correctly and in the correct directory.\n" << style::reset();
      param_args->help = 2;
    }
    if( a_graph.num_vertices() == 0){
      std::cout << style::apply(Color::red) << "Error: Cell graph from " + string_file + " is invalid. Make sure the graph is declared correctly.\n" << style::reset();
      param_args->help = 2;
    }
  } else {
    create_default_graph(&a_graph, cell_total, tissue_width);
  }
  param_args->adj_graph = std::move(a_graph);
}

}

#endif
