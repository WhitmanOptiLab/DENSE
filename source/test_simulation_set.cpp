#include "arg_parse.hpp"
#include "simulation_set.hpp"
#include "model_impl.hpp"
#include "context_impl.hpp"
#include <iostream>

int main(int argc, char *argv[])
{
    arg_parse::init(argc, argv);
    
    simulation_set sim_set(
        arg_parse::get<bool>("g", false, "gradients"),
        arg_parse::get<bool>("p", false, "perturb"),
        arg_parse::get<string>("mf", "../models/her_model_2014/param_list.csv", "model_file"),
        arg_parse::get<int>("ct", 200, "cell_total"),
        arg_parse::get<int>("tw", 50, "total_width"),
        arg_parse::get<RATETYPE>("ss", 0.01, "step_size") );
    sim_set.simulate_sets(arg_parse::get<int>("t", 60, "time"));
}
