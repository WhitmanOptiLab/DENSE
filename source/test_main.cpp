#include "arg_parse.hpp"
#include "color.hpp"
#include "simulation_set.hpp"
#include "model_impl.hpp"
#include "context_impl.hpp"
#include <iostream>


int main(int argc, char *argv[])
{
    arg_parse::init(argc, argv);
    color::enable(arg_parse::get<bool>("C", "no-color", true));
    
    if (arg_parse::get<bool>("H", "help") || arg_parse::get<bool>("h", "usage") || argc == 1)
    {
        // # Display all possible command line arguments with descriptions
        cout << color::set(color::YELLOW) <<
            "[-H | -h | --help | --usage]    " << color::set(color::GREEN) <<
            "Print information about program's various command line arguments." << color::clear() << endl;
        cout << color::set(color::YELLOW) <<
            "[-C | --no-color]               " << color::set(color::GREEN) <<
            "Disable color in the terminal." << color::clear() << endl;
            
            
        // TODO Improve these two help dialogues
        cout << color::set(color::YELLOW) <<
            "[-G | --gradients]              " << color::set(color::RED) <<
            "Enable {TODO: WRITE DESC}" << color::clear() << endl;
        cout << color::set(color::YELLOW) <<
            "[-P | --perturb]                " << color::set(color::RED) <<
            "Enable {TODO: WRITE DESC}" << color::clear() << endl;
            
            
        cout << color::set(color::YELLOW) <<
            "[-p | --param-list]    <string> " << color::set(color::GREEN) <<
            "Relative file location of the parameter list csv. \"../param_list.csv\", for example, excluding quotation marks." << color::clear() << endl;
        cout << color::set(color::YELLOW) <<
            "[-c | --cell-total]       <int> " << color::set(color::GREEN) <<
            "Total number of cells to simulate." << color::clear() << endl;
        cout << color::set(color::YELLOW) <<
            "[-w | --total-width]      <int> " << color::set(color::GREEN) <<
            "Width of tissue to simulate. Height is inferred by c/w." << color::clear() << endl;
        cout << color::set(color::YELLOW) <<
            "[-s | --step-size]   <RATETYPE> " << color::set(color::GREEN) <<
            "Increment size in which the simulation progresses through time." << color::clear() << endl;
        cout << color::set(color::YELLOW) <<
            "[-t | --time]             <int> " << color::set(color::GREEN) <<
            "Amount of time to simulate." << color::clear() << endl;
    }
    else
    {
        //arg_parse::get<RATETYPE>("i", 0.1, "analysis-interval");
        
        simulation_set sim_set(
            arg_parse::get<bool>("G", "gradients"),
            arg_parse::get<bool>("P", "perturb"),
            arg_parse::get<string>("p", "param-list"),
            arg_parse::get<int>("c", "cell-total"),
            arg_parse::get<int>("w", "total-width"),
            arg_parse::get<RATETYPE>("s", "step-size"),
            arg_parse::get<RATETYPE>("a", "analysis_interval", 10),
            arg_parse::get<RATETYPE>("t", "sim_time", 60));
        sim_set.simulate_sets();
        
    }
}
