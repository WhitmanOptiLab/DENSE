//
//  Callback.hpp
//
//
//  Created by Myan Sudharsanan on 7/9/20.
//

#ifndef Callback_h
#define Callback_h
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

template <typename Simulation>
class Callback{
    public:
        Callback(std::unique_ptr<Analysis<Simulation>> analysis, Simulation * simulation, csvw log) noexcept : analysis   { std::move(analysis) }, simulation(simulation), log { std::move(log) }
        {
        };

        std::unique_ptr<Analysis<Simulation>> get_analysis(){
            return std::move(analysis);
        }

        Details get_details(){
            return analysis->get_details();
        }

        void finalize(){
            analysis->finalize();
        }

        const Simulation* get_simulation() const{
            return simulation;
        }

        void operator()(){
            analysis->when_updated_by(*simulation, log.stream());
        }

        virtual ~Callback() = default;
        Callback(Callback&&)  = default;
        Callback & operator= (Callback&&) = default;

    private:
        std::unique_ptr<Analysis<Simulation>> analysis;
        Simulation* simulation;
        csvw log;

};
#endif
