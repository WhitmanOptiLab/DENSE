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
#include "sim/stoch/gillespie_direct_simulation.hpp"
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
using dense::parse_static_args;
using dense::parse_analysis_entries;
using dense::Static_Args;
using dense::run_and_return_analyses;
using dense::Details;
int printing_precision = 6;

std::vector<double> her2014_scorer (const std::vector<Parameter_Set>& population, Real real_results[], Sim_Builder<Fast_Gillespie_Direct_Simulation> sim, Static_Args a, int ac, char* av[]) {
	
	
	using Simulation = Fast_Gillespie_Direct_Simulation;
	std::vector<double> scores;
	
	for(auto param_set : population){
		
			auto simulation_means = run_and_return_analyses<Simulation>(a.simulation_duration, a.analysis_interval, std::move(sim.get_simulations(param_set)),parse_analysis_entries<Simulation>(ac, av, a.cell_total));
			
			double score  = 0.0;
			for(std::size_t j = 0;  j < simulation_means.size(); j++){
		      score += (((simulation_means[j]-real_results[j])*(simulation_means[j]-real_results[j]))+100);
					scores.push_back(100/score);
			}
		
	}
	
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
	// Initialize MPI if compiled with it
	#if defined(MPI)
		MPI_Init(&argc, &argv);
	#endif
  arg_parse::init(argc, argv);

  int popc = arg_parse::get<int>("pp", "population", 400);

  int miu = arg_parse::get<int>("m", "parents", NUM_PARAMS);

  std::string boundsfile;

  if (!arg_parse::get<std::string>("bb", "param-bounds", &boundsfile, true)) {
    return -1;
  }
  std::cout << boundsfile << '\n';

  csvr csv_in(boundsfile);

  Parameter_Set lBounds, uBounds;
		
  if (!lBounds.import_from(csv_in) || !uBounds.import_from(csv_in)) {
    std::cout << style::apply(Color::red) << "ERROR, parameter bounds file does not contain precisely two sets\n" << style::reset();
  }
	
	std::string real_inputs;
	if(!arg_parse::get<std::string>("ri", "real-input", &real_inputs,true)) {
		return -1;
	}
	std::cout << real_inputs << '\n';
	
	csvr csv_in_(real_inputs);
	
	Parameter_Set rinput;
	if (!rinput.import_from(csv_in_)){
		std::cout << style::apply(Color::red) << "Error, user inputs are misformatted \n" <<  style::reset();
	}
	
	//Init Simulation Object
	Static_Args args = parse_static_args(argc, argv);
	if(args.help == 1){
    return EXIT_SUCCESS;
  }
  if(args.help == 2){
    return EXIT_FAILURE;
  }
	
	 using Simulation = Fast_Gillespie_Direct_Simulation;
  Sim_Builder<Simulation> sim = Sim_Builder<Simulation>(args.perturbation_factors, args.gradient_factors, args.cell_total, args.tissue_width, argc, argv); 
	
  auto SRESscorer = 
		[&](const std::vector<Parameter_Set>& population) {
	    return her2014_scorer(population, rinput.data(), sim, args, argc, argv);
	  };
	
  SRES sres_driver(popc, miu, arg_parse::get<int>("nn", "num-generations", 100), lBounds, uBounds, SRESscorer);

	// Run libSRES
  for (int n = 1; n < arg_parse::get<int>("nn", "num-generations", 100); ++n) {
      sres_driver.nextGeneration();
  }

	return 0;
}

#if 0
/* usage prints the usage information and, optionally, an error message and then exits
	parameters:
		message: an error message to print before the usage information (set message to NULL or "\0" to not print any error)
	returns: nothing
	notes:
		This function exits after printing the usage information.
		Note that accept_input_params in init.cpp handles actual command-line input and that this information should be updated according to that function.
	todo:
		TODO somehow free memory even with the abrupt exit
*/
void usage (const char* message) {
	cout << endl;
	bool error = message != NULL && message[0] != '\0';
	if (error) {
		cout << term->red << message << term->reset << endl << endl;
	}
	cout << "Usage: [-option [value]]. . . [--option [value]]. . ." << endl;
	cout << "-r, --ranges-file        [filename]   : the relative filename of the ranges input file, default=none" << endl;
	cout << "-f, --simulation         [filename]   : the relative filename of the simulation executable, default=../simulation/simulation" << endl;
	cout << "-d, --dimensions         [int]        : the number of dimensions (i.e. rate parameters) to explore, min=1, default=45" << endl;
	cout << "-P, --parent-population  [int]        : the population of parent simulations to use each generation, min=1, default=3" << endl;
	cout << "-p, --total-population   [int]        : the population of total simulations to use each generation, min=1, default=20" << endl;
	cout << "-g, --generations        [int]        : the number of generations to run before returning results, min=1, default=1750" << endl;
	cout << "-s, --seed               [int]        : the seed used in the evolutionary strategy (not simulations), min=1, default=time" << endl;
	cout << "-e, --printing-precision [int]        : how many digits of precision parameters should be printed with, min=1, default=6" << endl;
	cout << "-a, --arguments          [N/A]        : every argument following this will be sent to the simulation" << endl;
	cout << "-c, --no-color           [N/A]        : disable coloring the terminal output, default=unused" << endl;
	cout << "-v, --verbose            [N/A]        : print detailed messages about the program state" << endl;
	cout << "-q, --quiet              [N/A]        : hide the terminal output, default=unused" << endl;
	cout << "-l, --licensing          [N/A]        : view licensing information (no simulations will be run)" << endl;
	cout << "-h, --help               [N/A]        : view usage information (i.e. this)" << endl;
	cout << endl << term->blue << "Example: ./sres-sampler " << term->reset << endl << endl;
	if (error) {
		exit(EXIT_INPUT_ERROR);
	} else {
		exit(EXIT_SUCCESS);
	}
}

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