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
#include "model_impl.hpp"
#include "io/ezxml/ezxml.h"
#include "arg_parse.hpp"
#include "Sim_Initializer.hpp"

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
using dense::conc_vector;

namespace dense {
    class Sim_Builder_Base {
      public:
        Sim_Builder_Base(Real* pf, Real** gf, NGraph::Graph adj_graph, int argc, char* argv[]) :
          perturbation_factors(pf), gradient_factors(gf)
        {
            arg_parse::init(argc, argv);
            using style::Mode;
            style::configure(arg_parse::get<bool>("n", "no-color", nullptr, false) ? Mode::disable : Mode::force);
            i_or_o = arg_parse::get<std::string>("d", "initial-conc", &init_conc, false);
            adjacency_graph = std::move(adj_graph);
            if (!arg_parse::get<int>("ff", "num-grow-cell", &num_grow_cell, false)) {
              num_grow_cell = 0;
            }
        }

      protected:
        Real* perturbation_factors;
        Real** gradient_factors;
        NGraph::Graph adjacency_graph;
        dense::Natural num_grow_cell;
        bool i_or_o;
        std::string init_conc;
    };

    class Sim_Builder_Determ : public Sim_Builder_Base {
      public:
        Sim_Builder_Determ(Real* pf, Real** gf, NGraph::Graph adj_graph, int argc, char* argv[]) :
          Sim_Builder_Base(pf, gf, adj_graph, argc, argv) 
        {
            step_size = arg_parse::get<Real>("s", "step-size", 0.0);
            //require step_size for deterministic simulation
            if(step_size == 0.0){
              arg_parse::get<bool>("s", "step-size", nullptr, true);
            }
            std::string init_conc;
            conc_vector(init_conc, i_or_o, &conc);
            adjacency_graph = std::move(adj_graph);
            if (!arg_parse::get<int>("ff", "num-grow-cell", &num_grow_cell, false)) {
              num_grow_cell = 0;
            }
        }
      protected:
        Real step_size;
        std::vector<Real> conc;
    };

    class Sim_Builder_Stoch : public Sim_Builder_Base {
      public:
        Sim_Builder_Stoch(Real* pf, Real** gf, NGraph::Graph adj_graph, int argc, char* argv[]) :
          Sim_Builder_Base(pf, gf, adj_graph, argc, argv) 
        {
            seed = 0;
            if (!arg_parse::get<int>("r", "rand-seed", &seed, false)) {
                seed = std::random_device()();
            }
            std::cout << "Stochastic simulation seed: " << seed << '\n';
            conc_vector(init_conc, i_or_o, &conc);
        }
      protected:
        int seed;
        std::vector<int> conc;
    };

    #ifndef __cpp_concepts
    template <class Simulation>
    #else
    template <Simulation_Concept Simulation>
    #endif
    class Sim_Builder {
        using This = Sim_Builder;

      public:
        Sim_Builder (This const&) = default;
        This& operator= (This&&);
        Sim_Builder<Simulation>(Real* pf, Real** gf, NGraph::Graph adj_graph, int argc, char* argv[]);
        std::vector<Simulation> get_simulations();
    };
}
#endif
