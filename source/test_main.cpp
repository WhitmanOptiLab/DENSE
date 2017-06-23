#include "arg_parse.hpp"
#include "analysis.hpp"
#include "color.hpp"
#include "csvr_sim.hpp"
#include "csvw_sim.hpp"
#include "datalogger.hpp"
#include "simulation_set.hpp"
#include "model_impl.hpp"
#include "context_determ.hpp"
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
            "[-G | --gradients]              " << color::set(color::GREEN) <<
            "Enable " << color::set(color::RED) <<
            "{TODO: WRITE DESC}" << color::clear() << endl;
        cout << color::set(color::YELLOW) <<
            "[-P | --perturb]                " << color::set(color::GREEN) <<
            "Enable " << color::set(color::RED) <<
            "{TODO: WRITE DESC}" << color::clear() << endl;
            
            
        cout << color::set(color::YELLOW) <<
            "[-p | --param-list]    <string> " << color::set(color::GREEN) <<
            "Relative file location and name of the parameter list csv. \"../param_list.csv\", for example, excluding quotation marks." << color::clear() << endl;
        cout << color::set(color::YELLOW) <<
            "[-e | --data-export]    <string> " << color::set(color::GREEN) <<
            "Relative file location and name of the output of the data logger csv. \"../data_out.csv\", for example, excluding quotation marks." << color::clear() << endl;
        cout << color::set(color::YELLOW) <<
            "[-i | --data-import]    <string> " << color::set(color::GREEN) <<
            "Relative file location and name of csv data to import into the data logger. \"../data_in.csv\", for example, excluding quotation marks. Using this flag skips the simulation." << color::clear() << endl;
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
            "[-a | --anlys-intvl]      <int> " << color::set(color::GREEN) <<
            "Analysis interval. How frequently data is fetched from simulation for analysis." << color::clear() << endl;
        cout << color::set(color::YELLOW) <<
            "[-r | --local_range] <RATETYPE> " << color::set(color::GREEN) <<
            "Range in which oscillation features are searched for." << color::clear() << endl;
        cout << color::set(color::YELLOW) <<
            "[-t | --time]             <int> " << color::set(color::GREEN) <<
            "Amount of time to simulate." << color::clear() << endl;
    }
    else
    {
        RATETYPE anlys_intvl = arg_parse::get<RATETYPE>("a", "anlys-intvl");
        
        simulation_set<simulation_determ> sim_set(
            arg_parse::get<bool>("G", "gradients"),
            arg_parse::get<bool>("P", "perturb"),
            arg_parse::get<string>("p", "param-list"),
            arg_parse::get<int>("c", "cell-total"),
            arg_parse::get<int>("w", "total-width"),
            arg_parse::get<RATETYPE>("s", "step-size"),
            anlys_intvl,
            arg_parse::get<RATETYPE>("t", "sim_time"));
        
        RATETYPE local_range = arg_parse::get<RATETYPE>("r", "local_range");
        string data_import = arg_parse::get<string>("i", "data-import", "");
        if (data_import.size() > 0)
        {
            csvr_sim csvrs(data_import);
            
            OscillationAnalysis oa(&csvrs, anlys_intvl, local_range, ph1);
            BasicAnalysis ba(&csvrs);
            
            csvrs.run();
            
            ba.test();
            oa.test();
        }
        else 
        {
            string data_export = arg_parse::get<string>("e", "data-export");
            if (data_export.size() > 0)
            {
                OscillationAnalysis oa(&sim_set._sim_set[0], anlys_intvl, local_range, ph1);
                BasicAnalysis ba(&sim_set._sim_set[0]);
                csvw_sim csvws(data_export, &sim_set._sim_set[0]);

                sim_set.simulate_sets();

                ba.test();
                oa.test();
            }
        }
    }
}
