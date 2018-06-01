#include "io/arg_parse.hpp"
#include "anlys/oscillation.hpp"
#include "anlys/basic.hpp"
#include "util/color.hpp"
#include "util/common_utils.hpp"
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
};

std::string file_add_num (
  std::string file_name, std::string const& prefix,
  char padding, unsigned file_no,
  std::size_t padded_size, std::string const& extension_sep)
{
  auto padded_file_no = left_pad(std::to_string(file_no), padded_size, padding);
  auto before_extension_sep = std::min(file_name.find_last_of(extension_sep), file_name.size());
  file_name.insert(before_extension_sep, prefix + padded_file_no);
  return file_name;
};

int main(int argc, char* argv[])
{
  arg_parse::init(argc, argv);
  color::enable(!arg_parse::get<bool>("n", "no-color", 0, false));

  if (arg_parse::get<bool>("h", "help", false) || arg_parse::get<bool>("H", "usage", false) || argc == 1) {
    // # Display all possible command line arguments with descriptions
    std::cout << color::set(color::YELLOW) <<
      "[-h | --help | --usage]         " << color::set(color::GREEN) <<
      "Print information about program's various command line arguments." << color::clear() << '\n';
    std::cout << color::set(color::YELLOW) <<
      "[-n | --no-color]               " << color::set(color::GREEN) <<
      "Disable color in the terminal." << color::clear() << '\n';
    std::cout << color::set(color::YELLOW) <<
      "[-p | --param-sets]    <string> " << color::set(color::GREEN) <<
      "Relative file location and name of the parameter sets csv. \"../param_sets.csv\", for example." << color::clear() << '\n';
    std::cout << color::set(color::YELLOW) <<
      "[-g | --gradients]     <string> " << color::set(color::GREEN) <<
      "Enables gradients and specifies the relative file location and name of the gradients csv. \"../param_grad.csv\", for example." << color::clear() << '\n';
    std::cout << color::set(color::YELLOW) <<
      "[-b | --perturbations] <string> " << color::set(color::GREEN) <<
      "Enables perturbations and specifies the relative file location and name of the perturbations csv. \"../param_pert.csv\", for example." << color::clear() << '\n';
    std::cout << color::set(color::YELLOW) <<
      "[-b|--perturbations] <RATETYPE> " << color::set(color::GREEN) <<
      "Enables perturbations and specifies a global perturbation factor to be applied to ALL reactions. The [-b | --perturb] flag itself is identical to the <string> version; the program automatically detects whether it is in the format of a file or a RATETYPE." << color::clear() << '\n';
    std::cout << color::set(color::YELLOW) <<
      "[-e | --data-export]   <string> " << color::set(color::GREEN) <<
      "Relative file location and name of the output of the logged data csv. \"../data_out.csv\", for example." << color::clear() << '\n';
    std::cout << color::set(color::YELLOW) <<
      "[-o | --specie-option] <string> " << color::set(color::GREEN) <<
      "Specify which species the logged data csv should output. This argument is only useful if [-e | --data-export] is being used. Not including this argument makes the program by default output/analyze all species. IF MORE THAN ONE BUT NOT ALL SPECIES ARE DESIRED, enclose the argument in quotation marks and seperate the species using commas. For example, \"alpha, bravo, charlie\", including quotation marks. If only one specie is desired, no commas or quotation marks are necessary." << color::clear() << '\n';
    std::cout << color::set(color::YELLOW) <<
      "[-i | --data-import]   <string> " << color::set(color::GREEN) <<
      "Relative file location and name of csv data to import into the analyses. \"../data_in.csv\", for example. Using this flag runs only analysis." << color::clear() << '\n';
    std::cout << color::set(color::YELLOW) <<
      "[-a | --analysis]      <string> " << color::set(color::GREEN) <<
      "Relative file location and name of the analysis config xml file. \"../analyses.xml\", for example. USING THIS ARGUMENT IMPLICITLY TOGGLES ANALYSIS." << color::clear() << '\n';
    std::cout << color::set(color::YELLOW) <<
      "[-c | --cell-total]       <int> " << color::set(color::GREEN) <<
      "Total number of cells to simulate." << color::clear() << '\n';
    std::cout << color::set(color::YELLOW) <<
      "[-w | --tissue-width]      <int> " << color::set(color::GREEN) <<
      "Width of tissue to simulate. Height is inferred by c/w." << color::clear() << '\n';
    std::cout << color::set(color::YELLOW) <<
      "[-r | --rand-seed]        <int> " << color::set(color::GREEN) <<
      "Set the stochastic simulation's random number generator seed." << color::clear() << '\n';
    std::cout << color::set(color::YELLOW) <<
      "[-s | --step-size]   <RATETYPE> " << color::set(color::GREEN) <<
      "Time increment by which the deterministic simulation progresses. USING THIS ARGUMENT IMPLICITLY SWITCHES THE SIMULATION FROM STOCHASTIC TO DETERMINISTIC." << color::clear() << '\n';
    std::cout << color::set(color::YELLOW) <<
      "[-t | --time-total]       <int> " << color::set(color::GREEN) <<
      "Amount of simulated minutes the simulation should execute." << color::clear() << '\n';
    std::cout << color::set(color::YELLOW) <<
      "[-u | --anlys-intvl] <RATETYPE> " << color::set(color::GREEN) <<
      "Analysis AND file writing time interval. How frequently (in units of simulated minutes) data is fetched from simulation for analysis and/or file writing." << color::clear() << '\n';
    std::cout << color::set(color::YELLOW) <<
      "[-v | --time-col]        <bool> " << color::set(color::GREEN) <<
      "Toggles whether file output includes a time column. Convenient for making graphs in Excel-like programs but slows down file writing. Time could be inferred without this column through the row number and the analysis interval." << color::clear() << '\n';
    std::cout << color::set(color::YELLOW) <<
      "[-N | --test-run]        <bool> " << color::set(color::GREEN) <<
      "Enables running a simulation without output for performance testing." << color::clear() << '\n';
    return EXIT_SUCCESS;
  }
  // Ambiguous "simulators" will either be a real simulations or input file
  vector<Observable*> simsAmbig;
  // We might do a sim_set, this is here just in case
  std::unique_ptr<simulation_set> simSet;

  // These fields are somewhat universal
  specie_vec default_specie_option;
  RATETYPE anlys_intvl, time_total;
  int cell_total;

  arg_parse::get<specie_vec>("o", "specie-option", &default_specie_option, false);
  // ========================== FILL SIMSAMBIG ==========================
  // See if importing
  // string is for either sim data import or export
  std::string data_ioe;
  if (arg_parse::get<std::string>("i", "data-import", &data_ioe, false))
  {
      simsAmbig.push_back(new csvr_sim(data_ioe, default_specie_option));
  }
  else // Not importing, do a real simulation
  {
      std::string param_sets;
      int tissue_width;

      // Required simulation fields
      if ( arg_parse::get<int>("c", "cell-total", &cell_total, true) &&
              arg_parse::get<int>("w", "tissue-width", &tissue_width, true) &&
              arg_parse::get<RATETYPE>("t", "time-total", &time_total, true) &&
              arg_parse::get<RATETYPE>("u", "anlys-intvl", &anlys_intvl, true) &&
              arg_parse::get<std::string>("p", "param-sets", &param_sets, true) )
      {
          // If step_size not set, create stochastic simulation
          RATETYPE step_size =
              arg_parse::get<RATETYPE>("s", "step-size", 0.0);
          int seed = arg_parse::get<int>("r", "rand-seed", time(0));

          // Warn user that they are not running deterministic sim
          if (step_size == 0.0)
          {
              std::cout << color::set(color::YELLOW) << "Running stochastic simulation. To run deterministic simulation, specify a step size using the [-s | --step-size] flag." << color::clear() << '\n';
              std::cout << "Stochastic simulation seed: " << seed << '\n';
          }

          std::vector<param_set> params;

          //Load parameter sets to run
          csvr_param csvrp(param_sets);
          for (unsigned int i=0; i<csvrp.get_total(); i++)
          {
            params.push_back(csvrp.get_next());
          }

          // Create simulation set
          simSet = std::unique_ptr<simulation_set>(new simulation_set(
            params,
            arg_parse::get<std::string>("g", "gradients", ""),
            arg_parse::get<std::string>("b", "perturbations", ""),
            cell_total, tissue_width, step_size,
            anlys_intvl, time_total, seed
          ));

          for (int i=0; i<simSet->getSetCount(); i++)
          {
              simsAmbig.push_back(simSet->_sim_set[i]);
          }
      }
  }


  // ========================== FILL ANLYSAMBIG =========================
  // Must have at least one observable
  if (simsAmbig.empty()) {
    std::cout << color::set(color::RED) << "Error: Your current set of command line arguments produces a useless state. (No inputs are specified.) Did you mean to use the [-i | --data-import] or the simulation-related flag(s)?" << color::clear() << '\n';
    return EXIT_FAILURE;
  }

  // Ambiguous "analyzers" will either be analyses or output file logs
  vector<Observer*> anlysAmbig;

  {
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
              for (int i=0; i<simsAmbig.size(); i++)
              {
                  RATETYPE
                      cell_start = strtol(ezxml_child(anlys,
                              "cell-start")->txt, 0, 0),
                      cell_end = strtol(ezxml_child(anlys,
                              "cell-end")->txt, 0, 0),
                      time_start = strtol(ezxml_child(anlys,
                              "time-start")->txt, 0, 0),
                      time_end = strtol(ezxml_child(anlys,
                              "time-end")->txt, 0, 0);
                  specie_vec specie_option = str_to_species(ezxml_child(anlys,
                          "species")->txt);

                  if (type == "basic")
                  {
                      std::string out_file =
                          ezxml_child(anlys, "out-file")->txt;

                      // If multiple sets, set file name
                      //   to "x_####.y"
                      csvw *csvwa = new csvw(
                              simsAmbig.size()==1 ?
                                  out_file :
                                  file_add_num(out_file,
                                      "_", '0', i, 4, "."));

                      anlysAmbig.push_back(new BasicAnalysis(
                                  simsAmbig[i], specie_option, csvwa,
                                  cell_start, cell_end,
                                  time_start, time_end));
                  }
                  else if (type == "oscillation")
                  {
                      RATETYPE win_range = strtol(ezxml_child(anlys, "win-range")->txt, 0, 0);
                      anlys_intvl = strtold(ezxml_child(anlys, "anlys-intvl")->txt, 0);
                      std::string out_file = ezxml_child(anlys, "out-file")->txt;

                      // If multiple sets, set file name
                      //   to "x_####.y"
                      csvw *csvwa = new csvw(
                              simsAmbig.size()==1 ?
                                  out_file :
                                  file_add_num(out_file,
                                      "_", '0', i, 4, "."));

                      anlysAmbig.push_back(new OscillationAnalysis(
                                  simsAmbig[i], anlys_intvl, win_range,
                                  specie_option, csvwa,
                                  cell_start, cell_end,
                                  time_start, time_end));
                  }
                  else
                  {
                      std::cout << color::set(color::YELLOW) << "Warning: No analysis type \"" << type << "\" found." << color::clear() << '\n';
                  }
              }
          }
      } else {// End analysis flag
          // Data export aka file log
          // Can't do it if already using data-import
          if (arg_parse::get<std::string>("e", "data-export", &data_ioe, false) &&
                          data_ioe != "")
          {
              for (unsigned int i=0; i<simsAmbig.size(); i++)
              {
                  // If multiple sets, set file name to "x_####.y"
                  csvw_sim *csvws = new csvw_sim(
                                  (simsAmbig.size()==1 ?
                                   data_ioe :
                                   file_add_num(data_ioe,
                                           "_", '0', i, 4, ".")),
                                  anlys_intvl, 0 /*time_start*/,
                                  time_total /*time_end*/,
                                  arg_parse::get<std::string>("v", "time-col", 0, false),
                                  cell_total, 0 /*cell_start*/,
                                  cell_total /*cell_end*/,
                                  default_specie_option, simsAmbig[i] );

                  anlysAmbig.push_back(csvws);
              }
          }
      }
      // End all observer preparation


      // ========================= RUN THE SHOW =========================
      // Only bother if there are outputs
      if (anlysAmbig.size() == 0)
      {
          if (!arg_parse::get<bool>("N", "test-run", 0, false)) {
              std::cout << color::set(color::YELLOW) << "Warning: performing basic analysis only.  Did you mean to use the [-e | --data-export] and/or [-a | --analysis] flag(s)? (use -N to suppress this error)" << color::clear() << '\n';
              for (int i=0; i<simsAmbig.size(); i++) {
                  anlysAmbig.push_back(new BasicAnalysis(
                                      simsAmbig[i], default_specie_option, NULL,
                                      0, cell_total,
                                      0, time_total));
              }
          }
      }

      for (auto simulation : simsAmbig) {
        simulation->run();
      }
  }

  // delete/write analyses
  for (auto anlys : anlysAmbig) {
      delete anlys;
  }

  // delete/write simulations
  for (auto sim : simsAmbig)
  {
      // Sometimes causes "munmap_chunck(): invalid pointer"
      // Honestly, this delete is not particularly important
      //if (sim) delete sim;
  }

  return EXIT_SUCCESS;
}
