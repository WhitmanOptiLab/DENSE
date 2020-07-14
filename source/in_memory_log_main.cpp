//
//  in_memory_log_main.cpp
//  
//
//  Created by Myan Sudharsanan on 7/2/20.
//
#ifndef LogTimes_main_h
#define LogTimes_main_h


#ifndef LogTimes_main_h
#define LogTimes_main_h
#include <stdio.h>
#include "io/arg_parse.hpp"
#include "measurement/oscillation.hpp"
#include "measurement/basic.hpp"
#include "measurement/bad_simulation_error.hpp"
#include "utility/style.hpp"
#include "utility/common_utils.hpp"
#include "io/csvr_sim.hpp"
#include "io/csvw_sim.hpp"
#include "sim/determ/num_sim.hpp"
#include "sim/determ/trap.hpp"
#include "sim/stoch/fast_gillespie_direct_simulation.hpp"
#include "sim/stoch/next_reaction_simulation.hpp"
#include "model_impl.hpp"
#include "io/ezxml/ezxml.h"
#include "Sim_Builder.hpp"
#include "run_simulation.hpp"
#include "run_analysis_only.hpp"
#include "arg_parse.hpp"
#include "parse_analysis_entries.hpp"
#include "io/csvr.hpp"
#include "sim/base.hpp"
#include "core/specie.hpp"
#include "in_memory_log.hpp"
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
using dense::csvw_sim;
using dense::Trapezoid_Simulation;
using dense::Sim_Builder;
using dense::parse_static_args;
using dense::parse_analysis_entries;
using dense::Static_Args;
using dense::run_simulation;
int main(int argc, char* argv[]){
    int ac = argc;
    char** av = argv;
    Static_Args args = parse_static_args(argc, argv);
    if(args.help == 1){
        return EXIT_SUCCESS;
    }
    if(args.help == 2){
        return EXIT_FAILURE;
    }
    using Simulation = Deterministic_Simulation;
    if(args.param_sets.size() == 0){
        std::cout << style::apply(Color::red) << "param_sets is empty" << '\n';
        return 0;
    }
    Sim_Builder<Simulation> sim = Sim_Builder<Simulation>(args.perturbation_factors, args.gradient_factors, args.adj_graph, ac, av);
    std::vector<Species> species_list;
#define SPECIE(name) species_list.push_back(#name);
#include "specie_list.hpp"
#undef SPECIE
    in_memory_log buffer = in_memory_log(species_list, {0,args.adj_graph.num_vertices()}, args.analysis_interval);
    std::vector<std::pair<std::string, std::unique_ptr<Analysis<Simulation>>>> buffer_analysis_form;
    buffer_analysis_form.emplace_back(out_file, std14::make_unique<in_memory_log<Simulation> >(
                                                                                               species_list, {0,args.adj_graph.num_vertices()}, args.analysis_interval));
    //vector of call backs or analysis, myans new function
    std::vector<Callback> in_memory_log_returns = run_analysis_only<Simulation>(args.simulation_duration, args.analysis_interval, sim.get_simulations(args.param_sets), buffer_analysis_form));
    //convert above vector to get vector of inmemory log objects which are the analyses?
    std::vector<in_memory_log> new_buffer;
    for(int i = 0; i < in_memory_log_returns.size(); i++){
        new_buffer.push_back(*(std::unique_ptr<in_memory_log>(in_memory_log_returns[i].get_analysis())));
    }
    run_simulation(args.simulation_duration, args.analysis_interval, new_buffer, parse_analysis_entries<Simulation>(argc, argv, args.adj_graph.num_vertices()))
}
#endif






















