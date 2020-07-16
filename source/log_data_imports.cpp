//
//  log_data_imports.cpp
//
//
//  Created by Myan Sudharsanan on 6/30/20.
//
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
#include "csvr.hpp"
#include "sim/base.hpp"
#include "core/specie.hpp"

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
    using Simulation = CSV_Streamed_Simulation;

    std::string data_import;
    Simulation s(arg_parse::get<std::string>("i", "data-import", &data_import));
    //Sim_Builder<Simulation> sim = Sim_Builder<Simulation>(args.perturbation_factors, args.gradient_factors, args.adj_graph, ac, av);
    run_analysis_only<Simulation>(args.simulation_duration, args.analysis_interval, s,parse_analysis_entries<Simulation>(argc, argv, args.adj_graph.num_vertices()));
}

#endif
