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
#include "io/csvr_param.hpp"
#include "util/common_utils.hpp"
#include "util/color.hpp"
#include "sim/set.hpp"
#include "sim/set.hpp"

#include <cstdlib>

using namespace std;

int printing_precision = 6;

std::vector<double> her2014_scorer (const vector<param_set>& population) {
  //Create mutants

  //Create simulations
  simulation_set(population,  
                 "", "",
                 arg_parse::get<int>("c", "cell-total", 200), 
                 arg_parse::get<int>("w", "tissue_width", 50),
                 arg_parse::get<RATETYPE>("s", "step-size", 0.01),
                 arg_parse::get<RATETYPE>("u", "anlys-intvl", 0.01),
                 arg_parse::get<int>("t", "time-total", 600),
                 arg_parse::get<int>("r", "rand-seed", time(0))
                );

  //Create analyses
  
  //Run simulation
  std::cout << "Running simulations!" << endl;

  //Calculate scores
  std::vector<double> scores;
  for (unsigned int i = 0; i < population.size(); ++i) {
    scores.push_back((double)rand() / (double)RAND_MAX);
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

  if (arg_parse::get<bool>("h", "help", false) || arg_parse::get<bool>("H", "usage", false) || argc == 1) {
    cout << color::set(color::YELLOW) << 
      "[-h | --help | --usage]         " << color::set(color::GREEN) << 
      "Print information about program's various command line arguments." 
      << color::clear() << endl;
    cout << color::set(color::YELLOW) << 
      "[-b | --param-bounds]   <string>" << color::set(color::GREEN) << 
      "Path to file for lower and upper bounds" 
      << color::clear() << endl;
    cout << color::set(color::YELLOW) << 
      "[-p | --population]        <int>" << color::set(color::GREEN) << 
      "Size of a population"
      << color::clear() << endl;
    cout << color::set(color::YELLOW) << 
      "[-m | --miu | --parents] <int>" << color::set(color::GREEN) << 
      "Size of the parent set between generations"
      << color::clear() << endl;
    cout << color::set(color::YELLOW) << 
      "[-n | --num-generations]   <int>" << color::set(color::GREEN) << 
      "Number of generations"
      << color::clear() << endl;
    return 0;
  } 
  int popc = arg_parse::get<int>("p", "population", 400);

  int miu = arg_parse::get<int>("m", "parents", NUM_PARAMS);

  string boundsfile;

  if (!arg_parse::get<string>("b", "param-bounds", &boundsfile, true)) {
    return -1;
  }
  std::cout << boundsfile << endl;

  csvr_param csvrp(boundsfile);

  if (csvrp.get_total() != 2) {
    cout << color::RED << "ERROR, parameter bounds file does not contain precisely two sets" << color::clear() << endl;
    return -1;
  }

  param_set lBounds = csvrp.get_next();
  param_set uBounds = csvrp.get_next();

  SRES sres_driver(popc, miu, arg_parse::get<int>("n", "num-generations", 100), lBounds, uBounds, her2014_scorer);

	// Run libSRES
  for (int n = 1; n < arg_parse::get<int>("n", "num-generations", 100); ++n) {
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

