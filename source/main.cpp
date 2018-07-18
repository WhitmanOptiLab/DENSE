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

using style::Color;

#include <cstdlib>
#include <random>
#include <memory>
#include <exception>
#include <iostream>

using dense::CSV_Streamed_Simulation;
using dense::Simulation_Set;
using dense::csvw_sim;
using dense::Deterministic_Simulation;
using dense::Stochastic_Simulation;

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

std::string xml_child_text(ezxml_t xml, char const* name, std::string default_ = "") {
  ezxml_t child = ezxml_child(xml, name);
  return child == nullptr ? default_ : child->txt;
}

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
    yellow << "[-N | --test-run]        <bool> " <<
    green << "Enables running a simulation without output for performance testing.\n" << style::reset();
}

int main(int argc, char* argv[]) {
  arg_parse::init(argc, argv);
  using style::Mode;
  style::configure(arg_parse::get<bool>("n", "no-color", nullptr, false) ? Mode::disable : Mode::force);

  if (arg_parse::get<bool>("h", "help", false) || arg_parse::get<bool>("H", "usage", false) || argc == 1) {
    display_usage(std::cout);
    return EXIT_SUCCESS;
  }
  // Ambiguous "simulators" will either be a real simulations or input file
  std::vector<std::unique_ptr<Simulation>> simulations;

  // These fields are somewhat universal
  specie_vec default_specie_option;
  Real anlys_intvl, time_total;
  int cell_total;

  arg_parse::get<specie_vec>("o", "specie-option", &default_specie_option, false);
  // ========================== FILL simulations ==========================
  // See if importing
  // string is for either sim data import or export
  std::string data_ioe;
  if (arg_parse::get<std::string>("i", "data-import", &data_ioe, false)) {
    simulations.emplace_back(new CSV_Streamed_Simulation(data_ioe, default_specie_option));
  }
  else {
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
          int seed = 0;
          if (step_size == 0.0) {
            if (!arg_parse::get<int>("r", "rand-seed", &seed, false)) {
              seed = std::random_device()();
            }
            // Warn user that they are not running deterministic sim
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

          Simulation_Set sim_set;

          if (step_size == 0.0) {
            Simulation_Set stochastic_set;
            for (auto& parameter_set : params) {
              stochastic_set.emplace<Stochastic_Simulation>(
                std::move(parameter_set), perturbation_factors, gradient_factors,
                cell_total, tissue_width, seed);
            }
            sim_set = std::move(stochastic_set);
          } else {
            Simulation_Set deterministic_set;
            for (auto& parameter_set : params) {
              deterministic_set.emplace<Deterministic_Simulation>(
                std::move(parameter_set), perturbation_factors, gradient_factors,
                cell_total, tissue_width, step_size);
            }
            sim_set = std::move(deterministic_set);
          }

          simulations.reserve(simulations.size() + sim_set.size());
          for (auto & sim : sim_set._sim_set) {
            simulations.emplace_back(sim);
          }
      }
      else {
        std::cout << style::apply(Color::red) <<
          "Error: Your current set of command line arguments produces a useless state. (No inputs are specified.) "
          "Did you mean to use the [-i | --data-import] or the simulation-related flag(s)?\n" << style::reset();
        return EXIT_FAILURE;
      }
  }


  // ========================== FILL ANLYSAMBIG =========================

  // Ambiguous "analyzers" will either be analyses or output file logs
  std::vector<std::pair<std::unique_ptr<Analysis>, std::size_t>> analysis_links;
  std::vector<std::unique_ptr<csvw>> csv_writers;

  // Analyses each with own file writer
  std::string config_file;
  if (arg_parse::get<std::string>("a", "analysis", &config_file, false)) {
    // Prepare analyses and writers
    ezxml_t config = ezxml_parse_file(config_file.c_str());

    for (ezxml_t anlys = ezxml_child(config, "anlys"); anlys != nullptr; anlys = anlys->next) {
      std::string type = ezxml_attr(anlys, "type");
      Real cell_start = std::stold(xml_child_text(anlys, "cell-start"));
      Real cell_end = std::stold(xml_child_text(anlys, "cell-end"));
      Real time_start = std::stold(xml_child_text(anlys, "time-start"));
      Real time_end = std::stold(xml_child_text(anlys, "time-end"));
      specie_vec specie_option = str_to_species(xml_child_text(anlys, "species"));
      std::string out_file = xml_child_text(anlys, "out-file");
      for (std::size_t i = 0; i < simulations.size(); ++i) {
        // If multiple sets, set file name to "x_####.y"
        csv_writers.emplace_back(new csvw(
          simulations.size() == 1 ? out_file : file_add_num(out_file, "_", '0', i, 4, ".")));

        if (type == "basic") {
          analysis_links.emplace_back(new BasicAnalysis(
            specie_option, cell_start, cell_end, time_start, time_end), i);
        }
        else if (type == "oscillation") {
          Real win_range = std::stold(xml_child_text(anlys, "win-range"));
          anlys_intvl = std::stold(xml_child_text(anlys, "anlys-intvl"));

          analysis_links.emplace_back(new OscillationAnalysis(
            anlys_intvl, win_range, specie_option, cell_start, cell_end, time_start, time_end), i);
        }
        else {
          std::cout << style::apply(Color::yellow) << "Warning: No analysis type \"" << type << "\" found.\n" << style::reset();
        }
      }
    }
  } else {// End analysis flag
    // Data export aka file log
    // Can't do it if already using data-import
    if (arg_parse::get<std::string>("e", "data-export", &data_ioe, false) && data_ioe != "") {
      for (std::size_t i = 0; i < simulations.size(); ++i) {
        new csvw_sim(
          (simulations.size() == 1 ? data_ioe : file_add_num(data_ioe, "_", '0', i, 4, ".")),
          arg_parse::get<std::string>("v", "time-col", nullptr, false),
          default_specie_option, *simulations[i] );
      }
    }
  }
  // End all observer preparation

  if (analysis_links.empty() && !arg_parse::get<bool>("N", "test-run", nullptr, false)) {
    std::cout << style::apply(Color::yellow) << "Warning: performing basic analysis only.  Did you mean to use the [-e | --data-export] and/or [-a | --analysis] flag(s)? (use -N to suppress this error)" << style::reset() << '\n';
    for (std::size_t i = 0; i < simulations.size(); ++i) {
      analysis_links.emplace_back(new BasicAnalysis(
        default_specie_option,
        0, cell_total, 0, time_total
      ), i);
    }
  }

  // ========================= RUN THE SHOW =========================

  auto duration = time_total;
  auto notify_interval = anlys_intvl;
  Real analysis_chunks = duration / notify_interval;
  int notifications_per_min = 1.0 / notify_interval;

  for (int a = 0; a < analysis_chunks; a++) {
    for (auto& analysis_link : analysis_links) {
      analysis_link.first->when_updated_by(*simulations[analysis_link.second]);
    }

    for (auto & simulation : simulations) {
      if (simulation->was_aborted()) continue;
      simulation->simulate_for(notify_interval);
      if (a % notifications_per_min == 0)
        std::cout << "Time: " << simulation->t << '\n';
    }
  }

  for (auto & simulation : simulations) {
    simulation->finalize();
  }
  for (auto& analysis_link : analysis_links) {
    analysis_link.first->finalize();
  }

  std::size_t i = 0;
  for (; i < csv_writers.size(); ++i) {
    analysis_links[i].first->show(csv_writers[i].get());
  }

  for(; i < analysis_links.size(); ++i) {
    analysis_links[i].first->show();
  }

  return EXIT_SUCCESS;
}

/*
Snapshot<> snapshot;
Snapshot<> data = simulation.snapshot();

template <typename Simulation>
Real Reaction_Traits<ph1_synthesis>::calculate_rate_for(Region<Simulation> region) {

}
*/
