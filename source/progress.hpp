//
//  progress.hpp
//
//
//  Created by Myan Sudharsanan on 6/24/20.
//

#ifndef progress_h
#define progress_h

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
using dense::Fast_Gillespie_Direct_Simulation;
using dense::stochastic::Next_Reaction_Simulation;
using dense::Details;
namespace dense {

    class Progress{
        private:
            std::string line_of_progress;
            dense::Natural n;
            Real end;
        public:
            Progress(std::string previous, dense::Natural a, Real limit){
                line_of_progress = previous;
                n = a;
                end = limit;
            }

            void set_line_of_progress(std::string current){
                line_of_progress = current;
            }

            void set_n(dense::Natural c){
                n = c;
            }

            void set_end(Real max){
                end = max;
            }

            void print_progress_bar(){
                string currline = "[";
                int barWidth = 70;
                //std::cout << "[";


                int loc = barWidth * (n / end);
                for (int i = 0; i < barWidth; ++i){
                    if (i < loc){
                        //std::cout << "=";
                        currline += "=";

                    } else if (i == loc){
                        //std::cout << ">";
                        currline += ">";
                    } else {
                        //std::cout << " ";
                        currline += " ";
                    }
                }
                currline += "] ";
                if (line_of_progress.compare(currline) != 0){

                    line_of_progress = currline;
                    std::cout << currline << int(100 * n / end) << " %\r";
                    std::cout.flush();
                    //std::cout << std::endl;
                }
            }
    };
}

#endif /* progress_h */
