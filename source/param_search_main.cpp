/*
Stochastically ranked evolutionary strategy sampler for zebrafish segmentation
Copyright (C) 2013 Ahmet Ay, Jack Holland, Adriana Sperlea, Sebastian Sangervasi
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
main.cpp contains the main, usage, and licensing functions.
Avoid putting functions in main.cpp that could be put in a more specific file.
*/

// Include MPI if compiled with it
#if defined(MPI)
	#undef MPI // MPI uses this macro as well, so temporarily undefine it
	#include <mpi.h> // Needed for MPI_Comm_rank, MPI_COMM_WORLD
	#define MPI // The MPI macro should be checked only for definition, not value
#endif

#if 0
#include "main.hpp" // Function declarations
#endif

#include "search/sres.hpp"
#include "io/arg_parse.hpp"
#include "io/csvr.hpp"
#include "utility/common_utils.hpp"
#include "utility/style.hpp"
#include "measurement/oscillation.hpp"
#include "measurement/basic.hpp"
#include "measurement/bad_simulation_error.hpp"
#include "io/csvr_sim.hpp"
#include "io/csvw_sim.hpp"
#include "sim/determ/determ.hpp"
#include "sim/stoch/fast_gillespie_direct_simulation.hpp"
#include "sim/stoch/next_reaction_simulation.hpp"
#include "model_impl.hpp"
#include "io/ezxml/ezxml.h"
#include "Sim_Builder.hpp"
#include "run_simulation.hpp"
#include "arg_parse.hpp"
#include "parse_analysis_entries.hpp"
#include "measurement/details.hpp"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cassert>
#include <random>
#include <memory>
#include <iterator>
#include <algorithm>
#include <functional>
#include <exception>
#include <stdexcept>


using style::Color;
using dense::csvw_sim;
using dense::Fast_Gillespie_Direct_Simulation;
using dense::Sim_Builder;
using dense::param_search_parse_static_args;
using dense::parse_analysis_entries;
using dense::Param_Static_Args;
using dense::run_and_return_analyses;
using dense::Details;
int printing_precision = 6;

std::vector<double> her2014_scorer (const std::vector<Parameter_Set>& population, std::vector<Real> real_results, Sim_Builder<Fast_Gillespie_Direct_Simulation> sim, Param_Static_Args a, const std::vector<std::pair<std::string, std::unique_ptr<Analysis<Fast_Gillespie_Direct_Simulation>>>> &analysisEntries) {
	
	using Simulation = Fast_Gillespie_Direct_Simulation;
	std::vector<double> scores;
	
//	for(auto param_set : population){
    
    auto simulation_means = run_and_return_analyses<Simulation>(a.simulation_duration, a.analysis_interval, std::move(sim.get_simulations(population)),analysisEntries);	
		
    double sse  = 0.0;	
    for(std::size_t j = 0;  j < simulation_means.size(); j++){
      
      sse += ((real_results[j] - simulation_means[j])*(real_results[j] -simulation_means[j]));
      
    }
   
    scores.push_back(sse);
//	}
	
  return scores;
}

/* main is called when the program is run and performs all program functionality
	parameters:
		argc: the number of command-line arguments
		argv: the array of command-line arguments
	returns: 0 on success, a positive integer on failure
	notes:
		Main should only delegate functionality; let the functions it calls handle specific tasks. This keeps the function looking clean and helps maintain the program structure.
	todo:
*/
int main (int argc, char** argv) {
	
	//Init Simulation Object
	Param_Static_Args args = param_search_parse_static_args(argc, argv);
	
  if(args.help == 1){
    return EXIT_SUCCESS;
  }
  
  if(args.help == 2){
    return EXIT_FAILURE;
  }
	
  using Simulation = Fast_Gillespie_Direct_Simulation;
  Sim_Builder<Simulation> sim = Sim_Builder<Simulation>(args.perturbation_factors, args.gradient_factors, args.adj_graph, argc, argv); 
	
  std::vector<std::pair<std::string, std::unique_ptr<Analysis<Simulation>>>> analysis_entries(std::move(parse_analysis_entries<Simulation>(argc, argv, args.adj_graph.num_vertices())));
	
  std::function<std::vector<Real>(const std::vector<Parameter_Set>&)> SRESscorer = 
    [&](const std::vector<Parameter_Set>& population) {
		
	    return her2014_scorer(population, args.real_input, sim, args, analysis_entries);
	  };
	
  SRES sres_driver(args.pop, args.parent, args.num_generations, args.lbounds, args.ubounds, SRESscorer);

	// Run libSRES
  for (int n = 1; n < args.num_generations; ++n) {
    sres_driver.nextGeneration();
  }

	return 0;
}

#if 0


/* licensing prints the program's copyright and licensing information and then exits
	parameters:
	returns: nothing
	notes:
	todo:
*/
void licensing () {
	cout << endl;
	cout << "Stochastically ranked evolutionary strategy sampler for zebrafish segmentation" << endl;
	cout << "Copyright (C) 2013 Ahmet Ay (aay@colgate.edu), Jack Holland (jholland@colgate.edu), Adriana Sperlea (asperlea@colgate.edu), Sebastian Sangervasi (ssangervasi@colgate.edu)" << endl;
	cout << "This program comes with ABSOLUTELY NO WARRANTY" << endl;
	cout << "This is free software, and you are welcome to redistribute it under certain conditions;" << endl;
	cout << "You can use this code and modify it as you wish under the condition that you refer to the article: \"Short-lived Her proteins drive robust synchronized oscillations in the zebrafish segmentation clock\" (Development 2013 140:3244-3253; doi:10.1242/dev.093278)" << endl;
	cout << endl;
	exit(EXIT_SUCCESS);
}
#endif