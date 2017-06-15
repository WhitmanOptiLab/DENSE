#include "arg_parse.hpp"
#include "simulation_set.hpp"
#include "model_impl.hpp"
#include "context_impl.hpp"
#include <iostream>

int main(int argc, char *argv[])
{
    arg_parse::init(argc, argv);
    

    simulation_set sim_set(
        arg_parse::get<bool>("G", "gradients", false),
        arg_parse::get<bool>("P", "perturb", false),
        arg_parse::get<string>("p", "param-list", "../param_list.csv"),
        arg_parse::get<int>("c", "cell-total", 200),
        arg_parse::get<int>("w", "total-width", 50),
        arg_parse::get<RATETYPE>("s", "step-size", 0.01),
        arg_parse::get<RATETYPE>("a", "analysis_interval", 10),
        arg_parse::get<RATETYPE>("t", "sim_time", 60) );
    sim_set.simulate_sets();
}
