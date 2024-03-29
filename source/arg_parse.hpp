#ifndef ARG_HPP
#define ARG_HPP

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
#include "Sim_Initializer.hpp"
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

using dense::csvw_sim;
using dense::CSV_Streamed_Simulation;
using dense::Deterministic_Simulation;
using dense::Fast_Gillespie_Direct_Simulation;
using dense::stochastic::Next_Reaction_Simulation;
using dense::graph_constructor;
using dense::Static_Args;
using dense::Param_Static_Args;

namespace dense{

void display_usage(std::ostream& out);
void display_usage(std::ostream& out) {
  auto yellow = style::apply(Color::yellow);
  auto green = style::apply(Color::green);
  out <<
    yellow << "[-h | --help | --usage]         " <<
    green << "Print information about program's various command line arguments.\n" <<
    yellow << "[-n | --no-color]               " <<
    green << "Disable color in the terminal.\n" <<
    yellow << "[-p | --param-sets]    <string> " <<
    green << "Relative file location and name of the parameter sets csv, e.g. '../param_sets.csv'.\n" <<
    yellow << "[-g | --gradients]     <string> " <<
    green << "Enables gradients and specifies the relative file location and name of the gradients csv, e.g. '../param_grad.csv'.\n" <<
    yellow << "[-b | --perturbations] <string> " <<
    green << "Enables perturbations and specifies the relative file location and name of the perturbations csv, e.g. '../param_pert.csv'.\n" <<
    yellow << "[-b|--perturbations]     <Real> " <<
    green << "Enables perturbations and specifies a global perturbation factor to be applied to ALL reactions. "
      "The [-b | --perturb] flag itself is identical to the <string> version; "
      "the program automatically detects whether it is in the format of a file or a real number.\n" <<
    yellow << "[-e | --data-export]   <string> " <<
    green << "Relative file location and name of the output of the logged data csv. \"../data_out.csv\", for example.\n" <<
    yellow << "[-o | --specie-option] <string> " <<
    green << "Specify which species the logged data csv should output. This argument is only useful if [-e | --data-export] is being used. "
      "Not including this argument makes the program by default output/analyze all species. "
      "IF MORE THAN ONE BUT NOT ALL SPECIES ARE DESIRED, enclose the argument in quotation marks and separate the species using commas. "
      "For example, \"alpha, bravo, charlie\", including quotation marks. If only one specie is desired, no commas or quotation marks are necessary." <<
    yellow << "[-i | --data-import]   <string> " <<
    green << "Relative file location and name of csv data to import into the analyses. \"../data_in.csv\", for example. Using this flag runs only analysis.\n" <<
    yellow << "[-a | --analysis]      <string> " <<
    green << "Relative file location and name of the analysis config xml file. \"../analyses.xml\", for example. USING THIS ARGUMENT IMPLICITLY TOGGLES ANALYSIS.\n" <<
    yellow << "[-c | --cell-total]       <int> " <<
    green << "Total number of cells to simulate.\n" <<
    yellow << "[-w | --tissue-width]      <int> " <<
    green << "Width of tissue to simulate. Height is inferred by c/w.\n" <<
    yellow << "[-r | --rand-seed]        <int> " <<
    green << "Set the stochastic simulation's random number generator seed.\n" <<
    yellow << "[-s | --step-size]       <Real> " <<
    green << "Time increment by which the deterministic simulation progresses. USING THIS ARGUMENT IMPLICITLY SWITCHES THE SIMULATION FROM STOCHASTIC TO DETERMINISTIC.\n" <<
    yellow << "[-t | --time-total]       <int> " <<
    green << "Amount of simulated minutes the simulation should execute.\n" <<
    yellow << "[-u | --anlys-intvl]     <Real> " <<
    green << "Analysis AND file writing time interval. How frequently (in units of simulated minutes) data is fetched from simulation for analysis and/or file writing.\n" <<
    yellow << "[-v | --time-col]        <bool> " <<
    green << "Toggles whether file output includes a time column. Convenient for making graphs in Excel-like programs but slows down file writing. Time could be inferred without this column through the row number and the analysis interval.\n" <<
    yellow << "[-d | --initial-conc]    <string>" <<
    green << "Relative file location and name of the file containing the initial concentrations of species.\n" <<
    yellow << "[-f | --cell-graph]    <string>" <<
    green << "Relative file location and name of the file containing the cell graph.\n" <<
    yellow << "[-N | --test-run]        <bool> " <<
    green << "Enables running a simulation without output for performance testing.\n" << style::reset();
}

Static_Args parse_static_args(int argc, char* argv[]){
Static_Args param_args;
param_args.help = 0;
arg_parse::init(argc, argv);
  using style::Mode;
  style::configure(arg_parse::get<bool>("n", "no-color", nullptr, false) ? Mode::disable : Mode::force);
  Real anlys_intvl, time_total;
  std::chrono::duration<Real, std::chrono::minutes::period> simulation_duration, analysis_interval;
  if (arg_parse::get<bool>("h", "help", false) || arg_parse::get<bool>("H", "usage", false) || argc == 1) {
    display_usage(std::cout);
    param_args.help = 1;
    return param_args;
  }
  int cell_total;
  int tissue_width;
  std::string param_sets;
  std::string cell_graph = "";

  // Required simulation fields
  if (!(arg_parse::get<Real>("t", "time-total", &time_total, true) &&
          arg_parse::get<Real>("u", "anlys-intvl", &anlys_intvl, true) &&
          arg_parse::get<std::string>("p", "param-sets", &param_sets, true))) {
    std::cout << style::apply(Color::red) <<
      "Error: Your current set of command line arguments produces a useless state. (No inputs are specified.) "
      "Did you mean to use the [-i | --data-import] or the simulation-related flag(s)?\n" << style::reset();
    param_args.help = 2;
    return param_args;
  }
  
  bool c_flag = arg_parse::get<int>("c", "cell-total", &cell_total, false);
  bool w_flag = arg_parse::get<int>("w", "tissue-width", &tissue_width, false);
  bool f_flag = arg_parse::get<std::string>("f", "cell-graph", &cell_graph, false);
  
  simulation_duration = decltype(simulation_duration)(time_total);
  analysis_interval = decltype(analysis_interval)(anlys_intvl);

  param_args.perturbation_factors = parse_perturbations(
    arg_parse::get<std::string>("b", "perturbations", ""));

  param_args.gradient_factors = parse_gradients(
    arg_parse::get<std::string>("g", "gradients", ""), tissue_width);
 
  param_args.simulation_duration = simulation_duration;
  param_args.analysis_interval = analysis_interval;
  param_args.param_sets =  csvr(param_sets).get_param_sets();
  
  if ( w_flag && c_flag && !f_flag){
    if (cell_total <= 0 || tissue_width <= 0){
      std::cout << style::apply(Color::red) <<
          "Error: Your current set of command line arguments produces a useless state. "
          "The cell total specified with [-c | --cell-total] or the tissue width specified with [-w | --tissue-width] is invalid.\n" << style::reset();
        param_args.help = 2;
        return param_args;
    }   
    if (tissue_width > cell_total){
      std::cout << style::apply(Color::red) <<
        "Error: Your current set of command line arguments produces a useless state. "
        "The tissue width specified with [-w | --tissue-width] cannot be larger than the cell total specified with [-c | --cell-total].\n" << style::reset();
      param_args.help = 2;
      return param_args;
    }
    if(tissue_width == 0){ tissue_width += 1; }
      graph_constructor(&param_args, "", cell_total, tissue_width);
  } else if ( f_flag && !c_flag && !w_flag) {
      graph_constructor(&param_args, cell_graph, 0, 0);
  } else {
      std::cout << style::apply(Color::red) <<
        "Error: Your current set of command line arguments produces a useless state. "
        "Either specify a cell total with [-c | --cell-total] and a width with [-w | --tissue-width] or provide a cell graph file with [-f | --cell-graph].\n" << style::reset();
      param_args.help = 2;
      return param_args;
  }
  return param_args;
}

	void display_param_search_usage(std::ostream& out);
void display_param_search_usage(std::ostream& out) {
  auto yellow = style::apply(Color::yellow);
  auto green = style::apply(Color::green);
  out <<
    yellow << "[-h | --help | --usage]         " <<
    green << "Print information about program's various command line arguments.\n" <<
    yellow << "[-n | --no-color]               " <<
    green << "Disable color in the terminal.\n" <<
    yellow << "[-g | --gradients]     <string> " <<
    green << "Enables gradients and specifies the relative file location and name of the gradients csv, e.g. '../param_grad.csv'.\n" <<
    yellow << "[-b | --perturbations] <string> " <<
    green << "Enables perturbations and specifies the relative file location and name of the perturbations csv, e.g. '../param_pert.csv'.\n" <<
    yellow << "[-b|--perturbations]     <Real> " <<
    green << "Enables perturbations and specifies a global perturbation factor to be applied to ALL reactions. "
      "The [-b | --perturb] flag itself is identical to the <string> version; "
      "the program automatically detects whether it is in the format of a file or a real number.\n" <<
    yellow << "[-e | --data-export]   <string> " <<
    green << "Relative file location and name of the output of the logged data csv. \"../data_out.csv\", for example.\n" <<
    yellow << "[-o | --specie-option] <string> " <<
    green << "Specify which species the logged data csv should output. This argument is only useful if [-e | --data-export] is being used. "
      "Not including this argument makes the program by default output/analyze all species. "
      "IF MORE THAN ONE BUT NOT ALL SPECIES ARE DESIRED, enclose the argument in quotation marks and separate the species using commas. "
      "For example, \"alpha, bravo, charlie\", including quotation marks. If only one specie is desired, no commas or quotation marks are necessary." <<
    yellow << "[-i | --data-import]   <string> " <<
    green << "Relative file location and name of csv data to import into the analyses. \"../data_in.csv\", for example. Using this flag runs only analysis.\n" <<
    yellow << "[-a | --analysis]      <string> " <<
    green << "Relative file location and name of the analysis config xml file. \"../analyses.xml\", for example. USING THIS ARGUMENT IMPLICITLY TOGGLES ANALYSIS.\n" <<
    yellow << "[-c | --cell-total]       <int> " <<
    green << "Total number of cells to simulate.\n" <<
    yellow << "[-w | --tissue-width]      <int> " <<
    green << "Width of tissue to simulate. Height is inferred by c/w.\n" <<
    yellow << "[-r | --rand-seed]        <int> " <<
    green << "Set the stochastic simulation's random number generator seed.\n" <<
    yellow << "[-s | --step-size]       <Real> " <<
    green << "Time increment by which the deterministic simulation progresses. USING THIS ARGUMENT IMPLICITLY SWITCHES THE SIMULATION FROM STOCHASTIC TO DETERMINISTIC.\n" <<
    yellow << "[-t | --time-total]       <int> " <<
    green << "Amount of simulated minutes the simulation should execute.\n" <<
    yellow << "[-u | --anlys-intvl]     <Real> " <<
    green << "Analysis AND file writing time interval. How frequently (in units of simulated minutes) data is fetched from simulation for analysis and/or file writing.\n" <<
    yellow << "[-v | --time-col]        <bool> " <<
    green << "Toggles whether file output includes a time column. Convenient for making graphs in Excel-like programs but slows down file writing. Time could be inferred without this column through the row number and the analysis interval.\n" <<
    yellow << "[-N | --test-run]        <bool> " <<
    green << "Enables running a simulation without output for performance testing.\n" << 
		yellow << "[-pp | --total-population]        <int> " <<
    green << "The population of total simulations to use each generation, min=1, default=400\n" << 
		yellow << "[-m| --parents]        <int> " <<
    green << "The number of parameters\n" << 
		yellow << "[-bb| --param-bounds]        <string> " <<
    green << "The filepath to CSV file with parameter search boundaries(requires precisely two sets)\n" << 
		yellow << "[-ri| --real-input]        <string> " <<
    green << "The filepath to a CSV file with the data which the estimated parameter simulations shall be scored against\n" <<
		yellow << "[-nn| --num-generations]        <int> " <<
    green << "The number of generations in which parameters shall be guessed.\n" << 
		style::reset();
}
	
Param_Static_Args param_search_parse_static_args(int argc, char* argv[]);

Param_Static_Args param_search_parse_static_args(int argc, char* argv[]){
Param_Static_Args param_args;
param_args.help = 0;
arg_parse::init(argc, argv);
  using style::Mode;
  style::configure(arg_parse::get<bool>("n", "no-color", nullptr, false) ? Mode::disable : Mode::force);
  Real anlys_intvl, time_total;
  std::chrono::duration<Real, std::chrono::minutes::period> simulation_duration, analysis_interval;
  if (arg_parse::get<bool>("h", "help", false) || arg_parse::get<bool>("H", "usage", false) || argc == 1) {
    display_param_search_usage(std::cout);
    param_args.help = 1;
    return param_args;
  }
  int cell_total;
  int tissue_width;
  std::string param_sets;
  std::string cell_graph = "";

  // Required simulation fields
  if (!(arg_parse::get<int>("c", "cell-total", &cell_total, true) &&
          arg_parse::get<int>("w", "tissue-width", &tissue_width, true) &&
          arg_parse::get<Real>("t", "time-total", &time_total, true) &&
          arg_parse::get<Real>("u", "anlys-intvl", &anlys_intvl, true)
         )) {
    std::cout << style::apply(Color::red) <<
      "Error: Your current set of command line arguments produces a useless state. (No inputs are specified.) "
      "Did you mean to use the [-i | --data-import] or the simulation-related flag(s)?\n" << style::reset();
    param_args.help = 2;
    return param_args;
  }
	int popc = arg_parse::get<int>("pp", "population", 400);

  int miu = arg_parse::get<int>("m", "parents", NUM_PARAMS);

  std::string boundsfile;

  if (!arg_parse::get<std::string>("bb", "param-bounds", &boundsfile, true)) {
    param_args.help = 2;
  }
  std::cout << boundsfile << '\n';

  csvr csv_in(boundsfile);

  std::vector<Parameter_Set> bounds = csv_in.get_param_sets();
		
  if (bounds.size() != 2) {
    std::cout << style::apply(Color::red) << "ERROR, parameter bounds file does not contain precisely two sets\n" << style::reset();
    param_args.help = 2;
  }
	
	std::string real_inputs;
	if(!arg_parse::get<std::string>("ri", "real-input", &real_inputs,true)) {
		param_args.help = 2;
	}
	std::cout << real_inputs << '\n';

	csvr csv_in_(real_inputs);
	std::vector<Real> rinput;
	Real entry = -1;
	Real *data_getter = &entry;

	
	if(!csv_in_.get_next(data_getter)){
		std::cout << style::apply(Color::red) << "Error, data submitted is misformatted \n" << style::reset();
	}

	std::cout << entry << '\n';
	rinput.push_back(entry);
	
	while(csv_in_.get_next(data_getter)){
		rinput.push_back(*data_getter);
	}

  simulation_duration = decltype(simulation_duration)(time_total);
  analysis_interval = decltype(analysis_interval)(anlys_intvl);

  param_args.perturbation_factors = parse_perturbations(
    arg_parse::get<std::string>("b", "perturbations", ""));

  param_args.gradient_factors = parse_gradients(
    arg_parse::get<std::string>("g", "gradients", ""), tissue_width);

  bool c_flag = arg_parse::get<int>("c", "cell-total", &cell_total, false);
  bool w_flag = arg_parse::get<int>("w", "tissue-width", &tissue_width, false);
  bool f_flag = arg_parse::get<std::string>("f", "cell-graph", &cell_graph, false);
  
  param_args.simulation_duration = simulation_duration;
  param_args.analysis_interval = analysis_interval;
  
  if ( w_flag && c_flag && !f_flag){
    if (cell_total <= 0 || tissue_width <= 0){
      std::cout << style::apply(Color::red) <<
          "Error: Your current set of command line arguments produces a useless state. "
          "The cell total specified with [-c | --cell-total] or the tissue width specified with [-w | --tissue-width] is invalid.\n" << style::reset();
        param_args.help = 2;
        return param_args;
    }   
    if (tissue_width > cell_total){
      std::cout << style::apply(Color::red) <<
        "Error: Your current set of command line arguments produces a useless state. "
        "The tissue width specified with [-w | --tissue-width] cannot be larger than the cell total specified with [-c | --cell-total].\n" << style::reset();
      param_args.help = 2;
      return param_args;
    }
    if(tissue_width == 0){ tissue_width += 1; }
      graph_constructor(&param_args, "", cell_total, tissue_width);
  } else if ( f_flag && !c_flag && !w_flag) {
      graph_constructor(&param_args, cell_graph, 0, 0);
  } else {
      std::cout << style::apply(Color::red) <<
        "Error: Your current set of command line arguments produces a useless state. "
        "Either specify a cell total with [-c | --cell-total] and a width with [-w | --tissue-width] or provide a cell graph file with [-f | --cell-graph].\n" << style::reset();
      param_args.help = 2;
      return param_args;
  }

	param_args.lbounds = bounds[0];
	param_args.ubounds = bounds[1];
	param_args.pop = popc;
	param_args.parent = miu;
	param_args.real_input = rinput;
	param_args.num_generations = arg_parse::get<int>("nn", "num-generations", 100);
  return param_args;
}

}

#endif
