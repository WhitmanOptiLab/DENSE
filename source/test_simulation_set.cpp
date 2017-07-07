#include "arg_parse.hpp"
#include "simulation_set.hpp"
#include "model_impl.hpp"
#include "context_determ.hpp"
#include <ctime>
#include <iostream>

int main(int argc, char *argv[])
{
    arg_parse::init(argc, argv);
    

    simulation_set sim_set(
        arg_parse::get<string>("g", "gradients", ""),
        arg_parse::get<string>("v", "perturb", ""),
        arg_parse::get<string>("p", "param-sets", "../param_sets.csv"),
        arg_parse::get<int>("c", "cell-total", 200),
        arg_parse::get<int>("w", "total-width", 50),
        // If step-size == 0, then it will be stochastic
        arg_parse::get<RATETYPE>("s", "step-size", 0.0),
        arg_parse::get<RATETYPE>("a", "analysis_interval", 10),
        arg_parse::get<RATETYPE>("t", "sim-time", 60),
        arg_parse::get<int>("r", "rand-seed", time(0)) );
    sim_set.simulate_sets();
}
