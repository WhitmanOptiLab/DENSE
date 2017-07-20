#include "io/arg_parse.hpp"
#include "anlys/oscillation.hpp"
#include "anlys/basic.hpp"
#include "util/color.hpp"
#include "util/common_utils.hpp"
#include "io/csvr_sim.hpp"
#include "io/csvw_sim.hpp"
#include "sim/set.hpp"
#include "model_impl.hpp"
#include "io/ezxml/ezxml.h"
#include <ctime>
#include <exception>
#include <iostream>


int main(int argc, char* argv[])
{
    arg_parse::init(argc, argv);
    color::enable(!arg_parse::get<bool>("n", "no-color", 0, false));
    
    if (arg_parse::get<bool>("h", "help", false) || arg_parse::get<bool>("H", "usage", false) || argc == 1)
    {
        // # Display all possible command line arguments with descriptions
        cout << color::set(color::YELLOW) <<
            "[-h | --help | --usage]         " << color::set(color::GREEN) <<
            "Print information about program's various command line arguments." << color::clear() << endl;
        cout << color::set(color::YELLOW) <<
            "[-n | --no-color]               " << color::set(color::GREEN) <<
            "Disable color in the terminal." << color::clear() << endl;
        cout << color::set(color::YELLOW) <<
            "[-p | --param-sets]    <string> " << color::set(color::GREEN) <<
            "Relative file location and name of the parameter sets csv. \"../param_sets.csv\", for example." << color::clear() << endl;
        cout << color::set(color::YELLOW) <<
            "[-g | --gradients]     <string> " << color::set(color::GREEN) <<
            "Enables gradients and specifies the relative file location and name of the gradients csv. \"../param_grad.csv\", for example." << color::clear() << endl;
        cout << color::set(color::YELLOW) <<
            "[-b | --perturbations] <string> " << color::set(color::GREEN) <<
            "Enables perturbations and specifies the relative file location and name of the perturbations csv. \"../param_pert.csv\", for example." << color::clear() << endl;
        cout << color::set(color::YELLOW) <<
            "[-b|--perturbations] <RATETYPE> " << color::set(color::GREEN) <<
            "Enables perturbations and specifies a global perturbation factor to be applied to ALL reactions. The [-b | --perturb] flag itself is identical to the <string> version; the program automatically detects whether it is in the format of a file or a RATETYPE." << color::clear() << endl;
        cout << color::set(color::YELLOW) <<
            "[-e | --data-export]   <string> " << color::set(color::GREEN) <<
            "Relative file location and name of the output of the logged data csv. \"../data_out.csv\", for example." << color::clear() << endl;
        cout << color::set(color::YELLOW) <<
            "[-o | --specie-option] <string> " << color::set(color::GREEN) <<
            "Specify which species the logged data csv should output. This argument is only useful if [-e | --data-export] is being used. Not including this argument makes the program by default output/analyze all species. IF MORE THAN ONE BUT NOT ALL SPECIES ARE DESIRED, enclose the argument in quotation marks and seperate the species using commas. For example, \"alpha, bravo, charlie\", including quotation marks. If only one specie is desired, no commas or quotation marks are necessary." << color::clear() << endl;
        cout << color::set(color::YELLOW) <<
            "[-i | --data-import]   <string> " << color::set(color::GREEN) <<
            "Relative file location and name of csv data to import into the analyses. \"../data_in.csv\", for example. Using this flag runs only analysis." << color::clear() << endl;
        cout << color::set(color::YELLOW) <<
            "[-a | --analysis]      <string> " << color::set(color::GREEN) <<
            "Relative file location and name of the analysis config xml file. \"../analyses.xml\", for example. USING THIS ARGUMENT IMPLICITLY TOGGLES ANALYSIS." << color::clear() << endl;
        cout << color::set(color::YELLOW) <<
            "[-c | --cell-total]       <int> " << color::set(color::GREEN) <<
            "Total number of cells to simulate." << color::clear() << endl;
        cout << color::set(color::YELLOW) <<
            "[-w | --tissue-width]      <int> " << color::set(color::GREEN) <<
            "Width of tissue to simulate. Height is inferred by c/w." << color::clear() << endl;
        cout << color::set(color::YELLOW) <<
            "[-r | --rand-seed]        <int> " << color::set(color::GREEN) <<
            "Set the stochastic simulation's random number generator seed." << color::clear() << endl;
        cout << color::set(color::YELLOW) <<
            "[-s | --step-size]   <RATETYPE> " << color::set(color::GREEN) <<
            "Time increment by which the deterministic simulation progresses. USING THIS ARGUMENT IMPLICITLY SWITCHES THE SIMULATION FROM STOCHASTIC TO DETERMINISTIC." << color::clear() << endl;
        cout << color::set(color::YELLOW) <<
            "[-t | --time-total]       <int> " << color::set(color::GREEN) <<
            "Amount of simulated minutes the simulation should execute." << color::clear() << endl;
        cout << color::set(color::YELLOW) <<
            "[-u | --anlys-intvl] <RATETYPE> " << color::set(color::GREEN) <<
            "Analysis AND file writing time interval. How frequently (in units of simulated minutes) data is fetched from simulation for analysis and/or file writing." << color::clear() << endl;
        cout << color::set(color::YELLOW) <<
            "[-v | --time-col]        <bool> " << color::set(color::GREEN) <<
            "Toggles whether file output includes a time column. Convenient for making graphs in Excel-like programs but slows down file writing. Time could be inferred without this column through the row number and the analysis interval." << color::clear() << endl;
    }
    else
    {
        // Ambiguous "simulators" will either be a real simulations or input file
        vector<Observable*> simsAmbig;
        // Ambiguous "analyzers" will either be analyses or output file logs
        vector<Observer*> anlysAmbig;
        // We might do a sim_set, this is here just in case
        simulation_set *simSet;
        
        // These fields are somewhat universal
        specie_vec specie_option;
        RATETYPE anlys_intvl, time_total;
        int cell_total;
        
        // ========================== FILL SIMSAMBIG ==========================
        // See if importing
        // string is for either sim data import or export
        string data_ioe = "";
        if (arg_parse::get<string>("i", "data-import", &data_ioe, false))
        {
            arg_parse::get<specie_vec>("o", "specie-option", &specie_option, false);
            simsAmbig.push_back(new csvr_sim(data_ioe, specie_option));
        }
        else // Not importing, do a real simulation
        {
            string param_sets;
            int tissue_width;

            // Required simulation fields
            if ( arg_parse::get<int>("c", "cell-total", &cell_total, true) &&
                    arg_parse::get<int>("w", "tissue-width", &tissue_width, true) &&
                    arg_parse::get<RATETYPE>("t", "time-total", &time_total, true) &&
                    arg_parse::get<RATETYPE>("u", "anlys-intvl", &anlys_intvl, true) &&
                    arg_parse::get<string>("p", "param-sets", &param_sets, true) )
            {
                // If step_size not set, create stochastic simulation
                RATETYPE step_size =
                    arg_parse::get<RATETYPE>("s", "step-size", 0.0);
                int seed = arg_parse::get<int>("r", "rand-seed", time(0));

                // Warn user that they are not running deterministic sim
                if (step_size == 0.0)
                {
                    cout << color::set(color::YELLOW) << "Running stochastic simulation. To run deterministic simulation, specify a step size using the [-s | --step-size] flag." << color::clear() << endl;
                    cout << "Stochastic simulation seed: " << seed << endl;
                }
                
                simSet = new simulation_set(param_sets,
                        arg_parse::get<string>("g", "gradients", ""),
                        arg_parse::get<string>("b", "perturbations", ""),
                        cell_total, tissue_width, step_size,
                        anlys_intvl, time_total, seed);
                
                for (int i=0; i<simSet->getSetCount(); i++)
                {
                    simsAmbig.push_back(simSet->_sim_set[i]);
                }
            }
        }
        
        
        // ========================== FILL ANLYSAMBIG =========================
        // Must have at least one observable
        if (simsAmbig.size() > 0)
        {
            // Data export aka file log
            // Can't do it if already using data-import
            if (arg_parse::get<string>("e", "data-export", &data_ioe, false) &&
                    data_ioe == "")
            {
                for (unsigned int i=0; i<simsAmbig.size(); i++)
                {
                    // If multiple sets, set file name to "x_####.y"
                    csvw_sim *csvws = new csvw_sim(
                            (simsAmbig.size()==1 ?
                                data_ioe :
                                file_add_num(data_ioe,
                                    "_", '0', i, 4, ".")),
                            anlys_intvl, 0 /*time_start*/,
                            time_total /*time_end*/,
                            arg_parse::get<string>("v", "time-col", 0, false),
                            cell_total, 0 /*cell_start*/,
                            cell_total /*cell_end*/,
                            specie_option, simsAmbig[i] );
                    
                    anlysAmbig.push_back(csvws);
                }
            }
            
            // Analyses each with own file writer
            string config_file;
            if (arg_parse::get<string>("a", "analysis", &config_file, false))
            {
                // Prepare analyses and writers
                ezxml_t config = ezxml_parse_file(config_file.c_str());

                for (ezxml_t anlys = ezxml_child(config, "anlys");
                        anlys; anlys = anlys->next)
                {
                    string type = ezxml_attr(anlys, "type");
                    for (int i=0; i<simsAmbig.size(); i++)
                    {
                        RATETYPE
                            cell_start = strtol(ezxml_child(anlys,
                                    "cell-start")->txt, 0, 0),
                            cell_end = strtol(ezxml_child(anlys,
                                    "cell-end")->txt, 0, 0),
                            time_start = strtol(ezxml_child(anlys,
                                    "time-start")->txt, 0, 0),
                            time_end = strtol(ezxml_child(anlys,
                                    "time-end")->txt, 0, 0);
                        specie_option = str_to_species(ezxml_child(anlys,
                                "species")->txt);

                        if (type == "basic")
                        {
                            string out_file =
                                ezxml_child(anlys, "out-file")->txt;

                            // If multiple sets, set file name
                            //   to "x_####.y"
                            csvw *csvwa = new csvw(
                                    simsAmbig.size()==1 ?
                                        out_file :
                                        file_add_num(out_file,
                                            "_", '0', i, 4, "."));
                            
                            anlysAmbig.push_back(new BasicAnalysis(
                                        simsAmbig[i], specie_option, csvwa,
                                        cell_start, cell_end,
                                        time_start, time_end));
                        }
                        else if (type == "oscillation")
                        {
                            RATETYPE win_range = strtol(ezxml_child(anlys, "win-range")->txt, 0, 0);
                            anlys_intvl = strtold(ezxml_child(anlys, "anlys-intvl")->txt, 0);
                            string out_file = ezxml_child(anlys, "out-file")->txt;
                            
                            // If multiple sets, set file name
                            //   to "x_####.y"
                            csvw *csvwa = new csvw(
                                    simsAmbig.size()==1 ?
                                        out_file :
                                        file_add_num(out_file,
                                            "_", '0', i, 4, "."));
                                            
                            anlysAmbig.push_back(new OscillationAnalysis(
                                        simsAmbig[i], anlys_intvl, win_range,
                                        specie_option, csvwa,
                                        cell_start, cell_end,
                                        time_start, time_end));
                        }
                        else
                        {
                            cout << color::set(color::YELLOW) << "Warning: No analysis type \"" << type << "\" found." << color::clear() << endl;
                        }
                    }
                }
            } // End analysis flag
            // End all observer preparation
            
            
            // ========================= RUN THE SHOW =========================
            // Only bother if there are outputs
            if (anlysAmbig.size() > 0)
            {
                for (int i=0; i<simsAmbig.size(); i++)
                {
                    simsAmbig[i]->run();
                }
            }
            else
            {
                cout << color::set(color::RED) << "Error: Your current set of command line arguments produces a useless state. (No outputs are being generated.) Did you mean to use the [-e | --data-export] and/or [-a | --analysis] flag(s)?" << color::clear() << endl;
            }
        }
        else // Error: no inputs
        {
            cout << color::set(color::RED) << "Error: Your current set of command line arguments produces a useless state. (No inputs are specified.) Did you mean to use the [-i | --data-import] or the simulation-related flag(s)?" << color::clear() << endl;
        } // End fill analyses and run
        
        // delete/write analyses
        for (auto anlys : anlysAmbig)
        {
            if (anlys) { delete anlys; }
        }
        
        // delete/write simulations
        for (auto sim : simsAmbig)
        {
            // Sometimes causes "munmap_chunck(): invalid pointer"
            // Honestly, this delete is not particularly important
            //if (sim) delete sim;
        }
        
        if (simSet) delete simSet;
    } // End -h|blank or not
}
