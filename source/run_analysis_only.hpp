//
//  run_analysis_only.hpp
//  
//
//  Created by Myan Sudharsanan on 6/26/20.
//

#ifndef run_analysis_only_h
#define run_analysis_only_h
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

namespace dense {
#ifndef __cpp_concepts
template <typename Simulation>
#else
template <Simulation_Concept Simulation>
#endif
std::vector<std::vector<Real>> run_analysis_only(std::chrono::duration<Real, std::chrono::minutes::period> duration, std::chrono::duration<Real, std::chrono::minutes::period> notify_interval, std::vector<Simulation> simulations,
const std::vector<std::pair<std::string, std::unique_ptr<Analysis<Simulation>>>> &analysis_entries);
    
    
#ifndef __cpp_concepts
template <typename Simulation>
#else
template <Simulation_Concept Simulation>
#endif
std::vector<std::vector<Real>> run_analysis_only(std::chrono::duration<Real, std::chrono::minutes::period> duration, std::chrono::duration<Real, std::chrono::minutes::period> notify_interval, std::vector<Simulation> simulations,
const std::vector<std::pair<std::string, std::unique_ptr<Analysis<Simulation>>>> &analysis_entries){
        
        
        struct Callback {
            Callback(
                     std::unique_ptr<Analysis<Simulation>> analysis,
                     Simulation & simulation,
                     csvw log
                     ) noexcept :
            analysis   { std::move(analysis) },
            simulation { std::addressof(simulation) },
            log        { std::move(log) }
            {
            }
            
            void operator()() {
                analysis->when_updated_by(*simulation, log.stream());
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
                callbacks.emplace_back(std::unique_ptr<Analysis<Simulation>>(name_and_analysis.second->clone()), simulations[i], out_file.empty() ? csvw(std::cout) : csvw(simulations.size() == 1 ? out_file : file_add_num(out_file, "_", '0', i, 4, ".")));
            }
        }
        // End all observer preparation
        
        // ========================= RUN THE SHOW =========================
        
        Real analysis_chunks = duration / notify_interval;
        int size_callbacks = callbacks.size();
    
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
                void age = simulation.age_by(notify_interval);
                int pos = (age / Minutes{1});
                pos = 0;
            }
        }
        
        std::vector<std::vector<Real>> analyses;
        size_t conc_size = callbacks[0].analysis->get_details().concs.size();
        
        
        for (auto& callback : callbacks) {
            callback.analysis->finalize();
            //     callback.analysis->show(&callback.log);
            
            Details analysis_details = callback.analysis->get_details();
            std::vector<Real> to_insert;
            for(size_t i = 0; i < analysis_details.concs.size(); i++){
                to_insert.push_back(analysis_details.concs[i]);
            }
            analyses.push_back(to_insert);
        }
        
        for (int i = analyses.size(); i < size_callbacks; i++){
            //std::vector<Real> to_insert(conc_size, 0);
            analyses.emplace_back(conc_size, 0);
            std::cout << "insert one \n";
        }
        return analyses;
        
    }
}

#endif /* run_analysis_only_h */

