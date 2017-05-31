#include "arg_parse.hpp"
#include "simulation_set.hpp"
#include "model_impl.hpp"
#include "context_impl.hpp"
#include <iostream>

int main(int argc, char *argv[])
{
    arg_parse::init(argc, argv);
    
    simulation_set sim_set(
        arg_parse::get<bool>("G", false, "gradients"),
        arg_parse::get<bool>("P", false, "perturb"),
        arg_parse::get<string>("p", "../models/her_model_2014/param_list.csv", "param-list"),
        arg_parse::get<int>("c", 200, "cell-total"),
        arg_parse::get<int>("w", 50, "total-width"),
        arg_parse::get<RATETYPE>("s", 0.01, "step-size") );
    sim_set.simulate_sets(arg_parse::get<int>("t", 60, "time"));
}
