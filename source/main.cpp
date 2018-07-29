#include "io/arg_parse.hpp"
#include "measurement/oscillation.hpp"
#include "measurement/basic.hpp"
#include "measurement/bad_simulation_error.hpp"
#include "utility/style.hpp"
#include "utility/common_utils.hpp"
#include "io/csvr_sim.hpp"
#include "io/csvw_sim.hpp"
#include "sim/determ/determ.hpp"
#include "sim/stoch/stoch.hpp"
#include "model_impl.hpp"
#include "io/ezxml/ezxml.h"

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
using dense::CSV_Streamed_Simulation;
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

#ifndef __cpp_concepts
template <typename Simulation>
#else
template <Simulation_Concept Simulation>
#endif
std::vector<std::pair<std::string, std::unique_ptr<Analysis<Simulation>>>> parse_analysis_entries();

#ifndef __cpp_concepts
template <typename Simulation>
#else
template <Simulation_Concept Simulation>
#endif
void run_simulation(
  std::chrono::duration<Real, std::chrono::minutes::period> duration,
  std::chrono::duration<Real, std::chrono::minutes::period> notify_interval,
  std::vector<Simulation> simulations,
  std::vector<std::pair<std::string, std::unique_ptr<Analysis<Simulation>>>> analysis_entries);

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

std::vector<Parameter_Set> parse_parameter_sets_csv(std::istream& in) {
  return { std::istream_iterator<Parameter_Set>(in), std::istream_iterator<Parameter_Set>() };
}

std::vector<Parameter_Set> parse_parameter_sets_csv(std::istream&& in) {
  return parse_parameter_sets_csv(in);
}

std::vector<Species> default_specie_option;
int cell_total;

int main(int argc, char* argv[]) {
  arg_parse::init(argc, argv);
  using style::Mode;
  style::configure(arg_parse::get<bool>("n", "no-color", nullptr, false) ? Mode::disable : Mode::force);

  if (arg_parse::get<bool>("h", "help", false) || arg_parse::get<bool>("H", "usage", false) || argc == 1) {
    display_usage(std::cout);
    return EXIT_SUCCESS;
  }
  // These fields are somewhat universal
  Real anlys_intvl, time_total;
  std::chrono::duration<Real, std::chrono::minutes::period> simulation_duration, analysis_interval;
  arg_parse::get<std::vector<Species>>("o", "specie-option", &default_specie_option, false);
  // ========================== FILL simulations ==========================
  // See if importing
  // string is for either sim data import or export
  std::string data_ioe;
  if (arg_parse::get<std::string>("i", "data-import", &data_ioe, false)) {
    using Simulation = CSV_Streamed_Simulation;
    std::vector<Simulation> simulations;
    simulations.emplace_back(data_ioe, default_specie_option);

    if (!(arg_parse::get<Real>("t", "time-total", &time_total, true) &&
    arg_parse::get<Real>("u", "anlys-intvl", &anlys_intvl, true))) {
      std::cerr << "No time-total or anlys-intvl for CSV Simulation (fix in future)\n";
      return EXIT_FAILURE;
    }
    simulation_duration = decltype(simulation_duration)(time_total);
    analysis_interval = decltype(analysis_interval)(anlys_intvl);
    run_simulation(simulation_duration, analysis_interval, std::move(simulations), parse_analysis_entries<Simulation>());
    return EXIT_SUCCESS;
  }
  std::string param_sets;
  int tissue_width;

  // Required simulation fields
  if (!(arg_parse::get<int>("c", "cell-total", &cell_total, true) &&
          arg_parse::get<int>("w", "tissue-width", &tissue_width, true) &&
          arg_parse::get<Real>("t", "time-total", &time_total, true) &&
          arg_parse::get<Real>("u", "anlys-intvl", &anlys_intvl, true) &&
          arg_parse::get<std::string>("p", "param-sets", &param_sets, true) )) {
    std::cout << style::apply(Color::red) <<
      "Error: Your current set of command line arguments produces a useless state. (No inputs are specified.) "
      "Did you mean to use the [-i | --data-import] or the simulation-related flag(s)?\n" << style::reset();
    return EXIT_FAILURE;
  }
  simulation_duration = decltype(simulation_duration)(time_total);
  analysis_interval = decltype(analysis_interval)(anlys_intvl);

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

  Real* perturbation_factors = parse_perturbations(
    arg_parse::get<std::string>("b", "perturbations", ""));

  Real** gradient_factors = parse_gradients(
    arg_parse::get<std::string>("g", "gradients", ""), tissue_width);

  auto parameter_sets = parse_parameter_sets_csv(std::ifstream(param_sets));

  if (step_size == 0.0) {
    using Simulation = Stochastic_Simulation;
    std::vector<Simulation> simulations;

    for (auto& parameter_set : parameter_sets) {
      simulations.emplace_back(
        std::move(parameter_set), perturbation_factors, gradient_factors,
        cell_total, tissue_width, seed);
    }

    run_simulation(simulation_duration, analysis_interval, std::move(simulations), parse_analysis_entries<Simulation>());
    return EXIT_SUCCESS;

  } else {
    using Simulation = Stochastic_Simulation;
    std::vector<Simulation> simulations;
    for (auto& parameter_set : parameter_sets) {
      simulations.emplace_back(
        std::move(parameter_set), perturbation_factors, gradient_factors,
        cell_total, tissue_width, step_size);
    }

    run_simulation(simulation_duration, analysis_interval, std::move(simulations), parse_analysis_entries<Simulation>());
    return EXIT_SUCCESS;
  }

}

#ifndef __cpp_concepts
template <typename Simulation>
#else
template <Simulation_Concept Simulation>
#endif
std::vector<std::pair<std::string, std::unique_ptr<Analysis<Simulation>>>> parse_analysis_entries() {

  decltype(parse_analysis_entries<Simulation>()) named_analysis_vector;

  // Analyses each with own file writer
  std::string config_file;
  std::string data_ioe;

  if (arg_parse::get<std::string>("a", "analysis", &config_file, false)) {
    // Prepare analyses and writers
    ezxml_t config = ezxml_parse_file(config_file.c_str());

    for (ezxml_t anlys = ezxml_child(config, "anlys"); anlys != nullptr; anlys = anlys->next) {
      std::string type = ezxml_attr(anlys, "type");
      std::pair<dense::Natural, dense::Natural> cell_range = {
        std::stold(xml_child_text(anlys, "cell-start")),
        std::stold(xml_child_text(anlys, "cell-end"))
      };
      std::pair<Real, Real> time_range = {
        std::stold(xml_child_text(anlys, "time-start")),
        std::stold(xml_child_text(anlys, "time-end"))
      };
      std::vector<Species> specie_option = str_to_species(xml_child_text(anlys, "species"));
      std::string out_file = xml_child_text(anlys, "out-file");

      if (type == "basic") {
        named_analysis_vector.emplace_back(out_file,
          std14::make_unique<BasicAnalysis<Simulation>>(
            specie_option, cell_range, time_range));
      }
      else if (type == "oscillation") {
        Real win_range = std::stold(xml_child_text(anlys, "win-range"));
        Real anlys_intvl = std::stold(xml_child_text(anlys, "anlys-intvl"));

        named_analysis_vector.emplace_back(out_file,
          std14::make_unique<OscillationAnalysis<Simulation>>(
            anlys_intvl, win_range, specie_option, cell_range, time_range));
      }
      else {
        std::cout << style::apply(Color::yellow) <<
          "Warning: Skipping unknown analysis type \"" << type << "\"\n" << style::reset();
      }
    }
  } else if (arg_parse::get<std::string>("e", "data-export", &data_ioe, false) && data_ioe != "") {
    // Data export aka file log
    // Can't do it if already using data-import
      named_analysis_vector.emplace_back(data_ioe, std14::make_unique<csvw_sim<Simulation>>(
        cell_total,
        arg_parse::get<std::string>("v", "time-col", nullptr, false),
        default_specie_option));
  }
  else if (!arg_parse::get<bool>("N", "test-run", nullptr, false)) {
      std::cout << style::apply(Color::yellow) << "Warning: performing basic analysis only.  Did you mean to use the [-e | --data-export] and/or [-a | --analysis] flag(s)? (use -N to suppress this error)" << style::reset() << '\n';

      named_analysis_vector.emplace_back("", std14::make_unique<BasicAnalysis<Simulation>>(
        default_specie_option, std::make_pair(0, cell_total)));
  }

  return named_analysis_vector;
}

#include <algorithm>
#include <utility>

#ifndef __cpp_concepts
template <typename Simulation>
#else
template <Simulation_Concept Simulation>
#endif
void run_simulation(
  std::chrono::duration<Real, std::chrono::minutes::period> duration,
  std::chrono::duration<Real, std::chrono::minutes::period> notify_interval,
  std::vector<Simulation> simulations,
  std::vector<std::pair<std::string, std::unique_ptr<Analysis<Simulation>>>> analysis_entries)
{

  struct Callback {

    Callback(
      std::unique_ptr<Analysis<Simulation>> analysis,
      Simulation & simulation,
      csvw log
    ):
      analysis   { std::move(analysis) },
      simulation { std::addressof(simulation) },
      log        { std::move(log) }
    {
    }

    void operator()() {
      return analysis->when_updated_by(*simulation, log.stream());
    }

    std::unique_ptr<Analysis<Simulation>> analysis;
    Simulation* simulation;
    csvw log;

  };

  std::vector<Callback> callbacks;
      // If multiple sets, set file name to "x_####.y"
  for (std::size_t i = 0; i < simulations.size(); ++i) {
    for (auto& name_and_analysis : analysis_entries) {
      auto& out_file = name_and_analysis.first;
      callbacks.emplace_back(
        std::unique_ptr<Analysis<Simulation>>(name_and_analysis.second->clone()),
        simulations[i],
        out_file.empty() ? csvw(std::cout) :
          csvw(simulations.size() == 1 ? out_file : file_add_num(out_file, "_", '0', i, 4, ".")));
    }
  }
  // End all observer preparation

  // ========================= RUN THE SHOW =========================

  Real analysis_chunks = duration / notify_interval;
  int notifications_per_min = decltype(duration)(1.0) / notify_interval;

  for (dense::Natural a = 0; a < analysis_chunks; a++) {
    std::vector<Simulation const*> bad_simulations;
    for (auto& callback : callbacks) {
      try {
        callback();
      }
      catch (dense::Bad_Simulation_Error<Simulation>& error) {
        bad_simulations.push_back(std::addressof(error.simulation()));
      }
    }
    for (auto& bad_simulation : bad_simulations) {
      auto has_bad_simulation = [=](Callback const& callback) {
        return callback.simulation == bad_simulation;
      };
      callbacks.erase(
        std::remove_if(callbacks.begin(), callbacks.end(), has_bad_simulation),
        callbacks.end());
      using std::swap;
      swap(simulations[bad_simulation - simulations.data()], simulations.back());
      simulations.pop_back();
    }

    for (auto & simulation : simulations) {
      simulation.simulate_for(notify_interval);
      if (a % notifications_per_min == 0) {
        std::cout << "Time: " << simulation.age().count() << '\n';
      }
    }
  }

  for (auto& callback : callbacks) {
    callback.analysis->finalize();
    callback.analysis->show(&callback.log);
  }

}
/*
Snapshot<> snapshot;
Snapshot<> data = simulation.snapshot();

template <typename Simulation>
Real Reaction_Traits<ph1_synthesis>::calculate_rate_for(Region<Simulation> region) {

}
*/
