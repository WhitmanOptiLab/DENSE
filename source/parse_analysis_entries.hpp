#ifndef  PARSE_HPP
#define  PARSE_HPP

#include "io/arg_parse.hpp"
#include "measurement/basic.hpp"
#include "measurement/convergence.hpp"
#include "measurement/oscillation.hpp"
#include "measurement/basic_oscillation.hpp"
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
using dense::Fast_Gillespie_Direct_Simulation;
using dense::stochastic::Next_Reaction_Simulation;
#include "measurement/perf.hpp"


namespace dense {

#ifndef __cpp_concepts
template <typename Simulation>
#else
template <Simulation_Concept Simulation>
#endif

std::string xml_child_text(ezxml_t xml, char const* name, std::string default_ = "");
std::string xml_child_text(ezxml_t xml, char const* name, std::string default_ = "") {
  ezxml_t child = ezxml_child(xml, name);
  return child == nullptr ? default_ : child->txt;
}

#ifndef __cpp_concepts
template <typename Simulation>
#else
template <Simulation_Concept Simulation>
#endif
std::vector<std::pair<std::string, std::unique_ptr<Analysis<Simulation>>>> parse_analysis_entries(int argc, char* argv[], int cell_total);

#ifndef __cpp_concepts
template <typename Simulation>
#else
template <Simulation_Concept Simulation>
#endif
std::vector<std::pair<std::string, std::unique_ptr<Analysis<Simulation>>>> parse_analysis_entries(int argc, char* argv[], int cell_total) {
  std::vector<std::pair<std::string, std::unique_ptr<Analysis<Simulation>>>>  named_analysis_vector;
	 arg_parse::init(argc, argv);
  using style::Mode;
  style::configure(arg_parse::get<bool>("n", "no-color", nullptr, false) ? Mode::disable : Mode::force);
  
  // Analyses each with own file writer
  std::string config_file;
  std::string data_ioe;
	 std::vector<Species> default_specie_option;
  arg_parse::get<std::vector<Species>>("o", "specie-option", &default_specie_option, false);

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
        
      } else if (type == "basic oscillation") {
          //Real win_range = std::stold(xml_child_text(anlys, "win-range"));
          Real anlys_intvl = std::stold(xml_child_text(anlys, "anlys-intvl"));
          
          named_analysis_vector.emplace_back(out_file,
            std14::make_unique<BasicOscillationAnalysis<Simulation>>(
              anlys_intvl, specie_option, cell_range, time_range));
      }
      else if (type == "convergence") {
        Real anlys_intvl = std::stold(xml_child_text(anlys, "anlys-intvl"));
        Real windowSize = std::stold(xml_child_text(anlys, "window-size"));
        Real thresHold = std::stold(xml_child_text(anlys, "threshold"));
          
        named_analysis_vector.emplace_back(out_file,
          std14::make_unique<ConvergenceAnalysis<Simulation>>(anlys_intvl, windowSize,
              thresHold, specie_option, cell_range, time_range));
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
  } else if (!arg_parse::get<bool>("N", "test-run", nullptr, false)) {
      std::cout << style::apply(Color::yellow) << "Warning: performing basic analysis only.  Did you mean to use the [-e | --data-export] and/or [-a | --analysis] flag(s)? (use -N to suppress this error)" << style::reset() << '\n';

      named_analysis_vector.emplace_back("", std14::make_unique<BasicAnalysis<Simulation>>(
        default_specie_option, std::make_pair(0, cell_total)));
  }
  named_analysis_vector.emplace_back("performance_obj", std14::make_unique<PerfAnalysis<Simulation>>(default_specie_option, std::make_pair(0, cell_total)) );

  return named_analysis_vector;
}
}
#endif
