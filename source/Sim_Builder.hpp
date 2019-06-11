#ifndef SIM_BUILDER_HPP
#define SIM_BUILDER_HPP

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
#include "arg_parse.hpp"

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
using dense::conc_vector;
namespace dense {

    #ifndef __cpp_concepts
    template <typename Simulation>
    #else
    template <Simulation_Concept Simulation>
    #endif
    class Sim_Builder {
        
        using This = Sim_Builder;

        public: 
            Sim_Builder (This const&) = default;
            This& operator= (This&&);
            Sim_Builder<Simulation>(Real* pf, Real** gf, int ct, int tw, int argc, char* argv[]);
            std::vector<Simulation> get_simulations();
        private:
            Real* perturbation_factors;
            Real** gradient_factors;
            int cell_total;
            int tissue_width;
   };

   template<>
   class Sim_Builder <Deterministic_Simulation>{
        using This = Sim_Builder<Deterministic_Simulation>;

        public: 
            This& operator= (This&&);
            Sim_Builder (This const&) = default;
            Sim_Builder(Real* pf, Real** gf, int ct, int tw, int argc, char* argv[]){
							     arg_parse::init(argc, argv);
        				using style::Mode;
        				style::configure(arg_parse::get<bool>("n", "no-color", nullptr, false) ? Mode::disable : Mode::force);
        				step_size = arg_parse::get<Real>("s", "step-size", 0.0);
        				perturbation_factors = pf;
       					gradient_factors = gf;
        				cell_total = ct;
       					tissue_width = tw;
            std::string init_conc;
            bool i_or_o = arg_parse::get<std::string>("d", "initial-conc", &init_conc, false);
            conc_vector(init_conc, i_or_o, &conc);
						}
      std::vector<Deterministic_Simulation> get_simulations(std::vector<Parameter_Set> param_sets){
							  std::vector<Deterministic_Simulation> simulations;
    					for (auto& parameter_set : param_sets) {
        					simulations.emplace_back(
          					std::move(parameter_set), perturbation_factors, gradient_factors,
          					cell_total, tissue_width, Minutes{step_size},conc);
      			}
   					return simulations;
						};
        private:
						Real* perturbation_factors;
						Real** gradient_factors;
						int cell_total;
						int tissue_width;
						Real step_size;
      std::vector<Real> conc;

   };
   template<>
   class Sim_Builder <Fast_Gillespie_Direct_Simulation>{
        using This = Sim_Builder<Fast_Gillespie_Direct_Simulation>;

        public: 
            Sim_Builder (This const&) = default;
            This& operator= (This&&);
            Sim_Builder(Real* pf, Real** gf, int ct, int tw, int argc, char* argv[]){
									arg_parse::init(argc, argv);
        					using style::Mode;
									style::configure(arg_parse::get<bool>("n", "no-color", nullptr, false) ? Mode::disable : Mode::force);
									seed = 0;
									if (!arg_parse::get<int>("r", "rand-seed", &seed, false)) {
									seed = std::random_device()();
									}
									// Warn user that they are not running deterministic sim
									std::cout << style::apply(Color::yellow) << "Running stochastic simulation. To run deterministic simulation, specify a step size using the [-s | --step-size] flag." << style::reset() << '\n';
									std::cout << "Stochastic simulation seed: " << seed << '\n';
									perturbation_factors = pf;
									gradient_factors = gf;
									cell_total = ct;
									tissue_width = tw;
         std::string init_conc;
         bool i_or_o = arg_parse::get<std::string>("d", "initial-conc", &init_conc, false);
         conc_vector(init_conc, i_or_o, &conc);
								}
           std::vector<Fast_Gillespie_Direct_Simulation> get_simulations(std::vector<Parameter_Set> param_sets){
									std::vector<Fast_Gillespie_Direct_Simulation> simulations;
									for (auto& parameter_set : param_sets) {
											simulations.emplace_back(
											std::move(parameter_set), perturbation_factors, gradient_factors,
											cell_total, tissue_width, seed, conc);
											}
									return simulations;
										}
        private:
						Real* perturbation_factors;
						Real** gradient_factors;
						int cell_total;
						int tissue_width;
						int seed;
      std::vector<int> conc;

   };
    template<>
   class Sim_Builder <Next_Reaction_Simulation>{
        using This = Sim_Builder<Next_Reaction_Simulation>;

        public: 
            Sim_Builder (This const&) = default;
            This& operator= (This&&);
            Sim_Builder(Real* pf, Real** gf, int ct, int tw, int argc, char* argv[]){
								arg_parse::init(argc, argv);
								using style::Mode;
								style::configure(arg_parse::get<bool>("n", "no-color", nullptr, false) ? Mode::disable : Mode::force);
								seed = 0;
								if (!arg_parse::get<int>("r", "rand-seed", &seed, false)) {
								seed = std::random_device()();
								}
								// Warn user that they are not running deterministic sim
								std::cout << style::apply(Color::yellow) << "Running stochastic simulation. To run deterministic simulation, specify a step size using the [-s | --step-size] flag." << style::reset() << '\n';
								std::cout << "Stochastic simulation seed: " << seed << '\n';
								perturbation_factors = pf;
								gradient_factors = gf;
								cell_total = ct;
								tissue_width = tw;
        std::string init_conc;
        bool i_or_o = arg_parse::get<std::string>("d", "initial-conc", &init_conc, false);
        conc_vector(init_conc, i_or_o, &conc);
								}
            std::vector<Next_Reaction_Simulation> get_simulations(std::vector<Parameter_Set> param_sets){
								std::vector<Next_Reaction_Simulation> simulations;
								for (auto& parameter_set : param_sets) {
								simulations.emplace_back(
								std::move(parameter_set), perturbation_factors, gradient_factors,
								cell_total, tissue_width, seed, conc);
				}
    						return simulations;
						}
        private:
						Real* perturbation_factors;
						Real** gradient_factors;
						int cell_total;
						int tissue_width;
						int seed;
      std::vector<int> conc;
   };
}
/*
Snapshot<> snapshot;
Snapshot<> data = simulation.snapshot();
template <typename Simulation>
Real Reaction_Traits<ph1_synthesis>::calculate_rate_for(Region<Simulation> region) {
}
*/
#endif