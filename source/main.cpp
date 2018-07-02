#include "io/arg_parse.hpp"
#include "measurement/oscillation.hpp"
#include "measurement/basic.hpp"
#include "utility/style.hpp"
#include "utility/common_utils.hpp"
#include "io/csvr_sim.hpp"
#include "io/csvw_sim.hpp"
#include "sim/set.hpp"
#include "model_impl.hpp"
#include "io/ezxml/ezxml.h"

#include <cstdlib>
#include <ctime>
#include <memory>
#include <exception>
#include <iostream>

std::string left_pad (std::string string, std::size_t min_size, char padding = ' ') {
  string.insert(string.begin(), min_size - std::min(min_size, string.size()), padding);
  return string;
}

std::string file_add_num (
  std::string file_name, std::string const& prefix,
  char padding, unsigned file_no,
  std::size_t padded_size, std::string const& extension_sep)
{
  auto padded_file_no = left_pad(std::to_string(file_no), padded_size, padding);
  auto before_extension_sep = std::min(file_name.find_last_of(extension_sep), file_name.size());
  file_name.insert(before_extension_sep, prefix + padded_file_no);
  return file_name;
}

int main(int argc, char* argv[])
{
  arg_parse::init(argc, argv);
  style::enable(!arg_parse::get<bool>("n", "no-color", nullptr, false));

  if (arg_parse::get<bool>("h", "help", false) || arg_parse::get<bool>("H", "usage", false) || argc == 1) {
    // # Display all possible command line arguments with descriptions
    std::cout << style::apply(Color::yellow) <<
      "[-h | --help | --usage]         " << style::apply(Color::green) <<
      "Print information about program's various command line arguments." << style::reset() << '\n';
    std::cout << style::apply(Color::yellow) <<
      "[-n | --no-color]               " << style::apply(Color::green) <<
      "Disable color in the terminal." << style::reset() << '\n';
    std::cout << style::apply(Color::yellow) <<
      "[-p | --param-sets]    <string> " << style::apply(Color::green) <<
      "Relative file location and name of the parameter sets csv. \"../param_sets.csv\", for example." << style::reset() << '\n';
    std::cout << style::apply(Color::yellow) <<
      "[-g | --gradients]     <string> " << style::apply(Color::green) <<
      "Enables gradients and specifies the relative file location and name of the gradients csv. \"../param_grad.csv\", for example." << style::reset() << '\n';
    std::cout << style::apply(Color::yellow) <<
      "[-b | --perturbations] <string> " << style::apply(Color::green) <<
      "Enables perturbations and specifies the relative file location and name of the perturbations csv. \"../param_pert.csv\", for example." << style::reset() << '\n';
    std::cout << style::apply(Color::yellow) <<
      "[-b|--perturbations]     <Real> " << style::apply(Color::green) <<
      "Enables perturbations and specifies a global perturbation factor to be applied to ALL reactions. The [-b | --perturb] flag itself is identical to the <string> version; the program automatically detects whether it is in the format of a file or a Real." << style::reset() << '\n';
    std::cout << style::apply(Color::yellow) <<
      "[-e | --data-export]   <string> " << style::apply(Color::green) <<
      "Relative file location and name of the output of the logged data csv. \"../data_out.csv\", for example." << style::reset() << '\n';
    std::cout << style::apply(Color::yellow) <<
      "[-o | --specie-option] <string> " << style::apply(Color::green) <<
      "Specify which species the logged data csv should output. This argument is only useful if [-e | --data-export] is being used. Not including this argument makes the program by default output/analyze all species. IF MORE THAN ONE BUT NOT ALL SPECIES ARE DESIRED, enclose the argument in quotation marks and seperate the species using commas. For example, \"alpha, bravo, charlie\", including quotation marks. If only one specie is desired, no commas or quotation marks are necessary." << style::reset() << '\n';
    std::cout << style::apply(Color::yellow) <<
      "[-i | --data-import]   <string> " << style::apply(Color::green) <<
      "Relative file location and name of csv data to import into the analyses. \"../data_in.csv\", for example. Using this flag runs only analysis." << style::reset() << '\n';
    std::cout << style::apply(Color::yellow) <<
      "[-a | --analysis]      <string> " << style::apply(Color::green) <<
      "Relative file location and name of the analysis config xml file. \"../analyses.xml\", for example. USING THIS ARGUMENT IMPLICITLY TOGGLES ANALYSIS." << style::reset() << '\n';
    std::cout << style::apply(Color::yellow) <<
      "[-c | --cell-total]       <int> " << style::apply(Color::green) <<
      "Total number of cells to simulate." << style::reset() << '\n';
    std::cout << style::apply(Color::yellow) <<
      "[-w | --tissue-width]      <int> " << style::apply(Color::green) <<
      "Width of tissue to simulate. Height is inferred by c/w." << style::reset() << '\n';
    std::cout << style::apply(Color::yellow) <<
      "[-r | --rand-seed]        <int> " << style::apply(Color::green) <<
      "Set the stochastic simulation's random number generator seed." << style::reset() << '\n';
    std::cout << style::apply(Color::yellow) <<
      "[-s | --step-size]       <Real> " << style::apply(Color::green) <<
      "Time increment by which the deterministic simulation progresses. USING THIS ARGUMENT IMPLICITLY SWITCHES THE SIMULATION FROM STOCHASTIC TO DETERMINISTIC." << style::reset() << '\n';
    std::cout << style::apply(Color::yellow) <<
      "[-t | --time-total]       <int> " << style::apply(Color::green) <<
      "Amount of simulated minutes the simulation should execute." << style::reset() << '\n';
    std::cout << style::apply(Color::yellow) <<
      "[-u | --anlys-intvl]     <Real> " << style::apply(Color::green) <<
      "Analysis AND file writing time interval. How frequently (in units of simulated minutes) data is fetched from simulation for analysis and/or file writing." << style::reset() << '\n';
    std::cout << style::apply(Color::yellow) <<
      "[-v | --time-col]        <bool> " << style::apply(Color::green) <<
      "Toggles whether file output includes a time column. Convenient for making graphs in Excel-like programs but slows down file writing. Time could be inferred without this column through the row number and the analysis interval." << style::reset() << '\n';
    std::cout << style::apply(Color::yellow) <<
      "[-N | --test-run]        <bool> " << style::apply(Color::green) <<
      "Enables running a simulation without output for performance testing." << style::reset() << '\n';
    return EXIT_SUCCESS;
  }
  // Ambiguous "simulators" will either be a real simulations or input file
  std::vector<std::unique_ptr<Simulation>> simsAmbig;

  // These fields are somewhat universal
  specie_vec default_specie_option;
  Real anlys_intvl, time_total;
  int cell_total;

  arg_parse::get<specie_vec>("o", "specie-option", &default_specie_option, false);
  // ========================== FILL SIMSAMBIG ==========================
  // See if importing
  // string is for either sim data import or export
  std::string data_ioe;
  if (arg_parse::get<std::string>("i", "data-import", &data_ioe, false))
  {
      simsAmbig.emplace_back(new CSV_Streamed_Simulation(data_ioe, default_specie_option));
  }
  else // Not importing, do a real simulation
  {
      std::string param_sets;
      int tissue_width;

      // Required simulation fields
      if ( arg_parse::get<int>("c", "cell-total", &cell_total, true) &&
              arg_parse::get<int>("w", "tissue-width", &tissue_width, true) &&
              arg_parse::get<Real>("t", "time-total", &time_total, true) &&
              arg_parse::get<Real>("u", "anlys-intvl", &anlys_intvl, true) &&
              arg_parse::get<std::string>("p", "param-sets", &param_sets, true) )
      {
          // If step_size not set, create stochastic simulation
          Real step_size = arg_parse::get<Real>("s", "step-size", 0.0);
          int seed = arg_parse::get<int>("r", "rand-seed", std::time(nullptr));

          // Warn user that they are not running deterministic sim
          if (step_size == 0.0) {
              std::cout << style::apply(Color::yellow) << "Running stochastic simulation. To run deterministic simulation, specify a step size using the [-s | --step-size] flag." << style::reset() << '\n';
              std::cout << "Stochastic simulation seed: " << seed << '\n';
          }

          std::vector<Parameter_Set> params;

          //Load parameter sets to run
          csvr csv_in(param_sets);
          Parameter_Set next_set;
          while (next_set.import_from(csv_in)) {
            params.push_back(next_set);
          }

          Real* perturbation_factors = parse_perturbations(
            arg_parse::get<std::string>("b", "perturbations", ""));

          Real** gradient_factors = parse_gradients(
            arg_parse::get<std::string>("g", "gradients", ""), tissue_width);

          // Create simulation set
          Simulation_Set sim_set(
            std::move(params),
            gradient_factors,
            perturbation_factors,
            cell_total, tissue_width, step_size,
            anlys_intvl, time_total, seed
          );

          simsAmbig.reserve(simsAmbig.size() + sim_set.size());
          for (auto & sim : sim_set._sim_set) {
            simsAmbig.emplace_back(sim);
          }
      }
  }


  // ========================== FILL ANLYSAMBIG =========================
  // Must have at least one observable
  if (simsAmbig.empty()) {
    std::cout << style::apply(Color::red) << "Error: Your current set of command line arguments produces a useless state. (No inputs are specified.) Did you mean to use the [-i | --data-import] or the simulation-related flag(s)?" << style::reset() << '\n';
    return EXIT_FAILURE;
  }

  // Ambiguous "analyzers" will either be analyses or output file logs
  std::vector<std::unique_ptr<Analysis>> anlysAmbig;
  std::vector<std::unique_ptr<csvw>> csv_writers;

  // Analyses each with own file writer
  std::string config_file;
  if (arg_parse::get<std::string>("a", "analysis", &config_file, false))
  {
      // Prepare analyses and writers
      ezxml_t config = ezxml_parse_file(config_file.c_str());

      for (ezxml_t anlys = ezxml_child(config, "anlys");
              anlys; anlys = anlys->next)
      {
          std::string type = ezxml_attr(anlys, "type");
          for (std::size_t i = 0; i < simsAmbig.size(); ++i)
          {
              Real
                  cell_start = strtol(ezxml_child(anlys,
                          "cell-start")->txt, nullptr, 0),
                  cell_end = strtol(ezxml_child(anlys,
                          "cell-end")->txt, nullptr, 0),
                  time_start = strtol(ezxml_child(anlys,
                          "time-start")->txt, nullptr, 0),
                  time_end = strtol(ezxml_child(anlys,
                          "time-end")->txt, nullptr, 0);
              specie_vec specie_option = str_to_species(ezxml_child(anlys,
                      "species")->txt);

              if (type == "basic")
              {
                  std::string out_file =
                      ezxml_child(anlys, "out-file")->txt;

                  // If multiple sets, set file name
                  //   to "x_####.y"
                  csv_writers.emplace_back(new csvw(simsAmbig.size()==1 ?
                              out_file :
                              file_add_num(out_file,
                                  "_", '0', i, 4, ".")));

                  anlysAmbig.emplace_back(new BasicAnalysis(
                              *simsAmbig[i], specie_option,
                              cell_start, cell_end,
                              time_start, time_end));
              }
              else if (type == "oscillation")
              {
                  Real win_range = strtol(ezxml_child(anlys, "win-range")->txt, nullptr, 0);
                  anlys_intvl = strtold(ezxml_child(anlys, "anlys-intvl")->txt, nullptr);
                  std::string out_file = ezxml_child(anlys, "out-file")->txt;

                  // If multiple sets, set file name
                  //   to "x_####.y"
                  csv_writers.emplace_back(new csvw(simsAmbig.size()==1 ?
                              out_file :
                              file_add_num(out_file,
                                  "_", '0', i, 4, ".")));

                  anlysAmbig.emplace_back(new OscillationAnalysis(
                              *simsAmbig[i], anlys_intvl, win_range,
                              specie_option,
                              cell_start, cell_end,
                              time_start, time_end));
              }
              else {
                std::cout << style::apply(Color::yellow) << "Warning: No analysis type \"" << type << "\" found." << style::reset() << '\n';
              }
          }
      }
  } else {// End analysis flag
    // Data export aka file log
    // Can't do it if already using data-import
    if (arg_parse::get<std::string>("e", "data-export", &data_ioe, false) && data_ioe != "")
    {
      for (std::size_t i = 0; i < simsAmbig.size(); ++i) {
        new csvw_sim(
          (simsAmbig.size() == 1 ? data_ioe : file_add_num(data_ioe, "_", '0', i, 4, ".")),
          anlys_intvl, 0 /*time_start*/,
          time_total /*time_end*/,
          arg_parse::get<std::string>("v", "time-col", nullptr, false),
          cell_total, 0 /*cell_start*/,
          cell_total /*cell_end*/,
          default_specie_option, *simsAmbig[i] );
      }
    }
  }
  // End all observer preparation


  // ========================= RUN THE SHOW =========================
  // Only bother if there are outputs
  if (anlysAmbig.empty() && !arg_parse::get<bool>("N", "test-run", nullptr, false)) {
    std::cout << style::apply(Color::yellow) << "Warning: performing basic analysis only.  Did you mean to use the [-e | --data-export] and/or [-a | --analysis] flag(s)? (use -N to suppress this error)" << style::reset() << '\n';
    for (auto & simulation : simsAmbig) {
      anlysAmbig.emplace_back(new BasicAnalysis(
        *simulation, default_specie_option,
        0, cell_total, 0, time_total
      ));
    }
  }

  for (auto & simulation : simsAmbig) {
    simulation->simulate();
    simulation->finalize();
  }

  std::size_t i = 0;
  for (; i < csv_writers.size(); ++i) {
    anlysAmbig[i]->show(csv_writers[i].get());
  }

  for(; i < anlysAmbig.size(); ++i) {
    anlysAmbig[i]->show();
  }


  return EXIT_SUCCESS;
}
