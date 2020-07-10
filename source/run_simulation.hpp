#ifndef RUN_SIM_HPP
#define RUN_SIM_HPP

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
#include "measurement/details.hpp"
#include "progress.hpp"
#include "Callback.hpp"

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
#include <utility>

using dense::csvw_sim;
using dense::CSV_Streamed_Simulation;
using dense::Deterministic_Simulation;
using dense::Fast_Gillespie_Direct_Simulation;
using dense::stochastic::Next_Reaction_Simulation;
using dense::Details;


//static bool check = false;
//static int num_equals = 0;
//static int prev_num_equals = 0;


//static int step = 1;
//static int displayNext = step;
//static int percent = 0;
//static char square = (char) 254;

//std::string left_pad (std::string string, std::size_t min_size, char padding = ' ');

std::string left_pad (std::string string, std::size_t min_size, char padding = ' ') {
  string.insert(string.begin(), min_size - std::min(min_size, string.size()), padding);
  return string;
}

/*std::string file_add_num (
  *std::string file_name, std::string const& prefix,
  *char padding, unsigned file_no,
  *std::size_t padded_size, std::string const& extension_sep);
*/
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
namespace dense {




std::string line_of_progress;


#ifndef __cpp_concepts
template <typename Simulation>
#else
template <Simulation_Concept Simulation>
#endif
std::vector<Callback> run_simulation(std::chrono::duration<Real, std::chrono::minutes::period> duration, std::chrono::duration<Real, std::chrono::minutes::period> notify_interval, std::vector<Simulation> simulations, std::vector<std::pair<std::string, std::unique_ptr<Analysis<Simulation>>>> analysis_entries);
    
    
#ifndef __cpp_concepts
template <typename Simulation>
#else
template <Simulation_Concept Simulation>
#endif
    std::vector<Callback> run_simulation(std::chrono::duration<Real, std::chrono::minutes::period> duration, std::chrono::duration<Real, std::chrono::minutes::period> notify_interval, std::vector<Simulation> simulations, std::vector<std::pair<std::string, std::unique_ptr<Analysis<Simulation>>>> analysis_entries){
        
        std::vector<Callback> callbacks;
        // If multiple sets, set file name to "x_####.y"
        for (std::size_t i = 0; i < simulations.size(); ++i) {
            for (auto& name_and_analysis : analysis_entries) {
                auto& out_file = name_and_analysis.first;
            callbacks.emplace_back(std::unique_ptr<Analysis<Simulation>>(name_and_analysis.second->clone()), simulations[i], out_file.empty() ? csvw(std::cout) :
                csvw(simulations.size() == 1 ? out_file : file_add_num(out_file, "_", '0', i, 4, ".")));
            }
        }
        // End all observer preparation
        
        // ========================= RUN THE SHOW =========================
        
        Real analysis_chunks = duration / notify_interval;
        int notifications_per_min = decltype(duration)(1.0) / notify_interval;
        
        dense::Natural a = 0;
        Progress p(line_of_progress, a, analysis_chunks);
        for (a = 0; a < analysis_chunks; a++) {
            p.set_line_of_progress(line_of_progress);
            p.set_n(a);
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
            
            if (a % notifications_per_min == 0) {
                //int pos = (age / Minutes{1});
                
                p.print_progress_bar();
            }
            for (auto & simulation : simulations) {
                (void) simulation.age_by(notify_interval);
            }
        }
        
        return callbacks;
        
        
        
    }
    

    
    
}
    



#endif
