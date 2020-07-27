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
#include "arg_parse.hpp"
#include "parse_analysis_entries.hpp"
#include "Callback.hpp"
#include "runtimecheck.hpp"

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
  using Simulation = dense::Trapezoid_Simulation;
  Sim_Builder<Simulation> sim = Sim_Builder<Simulation>(args.perturbation_factors, args.gradient_factors, args.adj_graph, ac, av);
    runtimecheck r;
    std::vector<Callback<Simulation> >  callbacks = run_simulation<Simulation>(args.simulation_duration, args.analysis_interval, sim.get_simulations(args.param_sets),parse_analysis_entries<Simulation>(argc, argv, args.adj_graph.num_vertices()));
    
    r.set_end();
    r.set_begin();
    
    for (auto& callback : callbacks) {
        callback.finalize();
        callback.show();
    }
    
    r.set_end();
    
    auto duration1 = r.get_duration(0, 0);
    std::cout << "Simulation only: " << duration1 << endl;
    auto duration2 = r.get_duration(1, 1);
    std::cout << "Finalization only: " << duration2 << endl;
    auto duration3 = r.get_duration(2, 0);
    std::cout << "Total duration: " << duration3 << endl;
}
/*
Snapshot<> snapshot;
Snapshot<> data = simulation.snapshot();
template <typename Simulation>
Real Reaction_Traits<ph1_synthesis>::calculate_rate_for(Region<Simulation> region) {
}
*/
