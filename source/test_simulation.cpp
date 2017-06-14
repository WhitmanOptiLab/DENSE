#include "arg_parse.hpp"
#include "simulation.hpp"
#include "datalogger.hpp"
#include "model_impl.hpp"
#include "context_impl.hpp"
#include "analysis.hpp"
#include "csvr_param.hpp"
#include <iostream>

int main(int argc, char *argv[]) {
    arg_parse::init(argc, argv);
    
    //setting up model
    model m(arg_parse::get<bool>("G", "gradients", false),
        arg_parse::get<bool>("P", "perturb", false));
    
    //setting up param_set
    param_set ps;
    csvr_param csvrp(arg_parse::get<string>("p", "param-list", "../models/her_model_2014/param_list.csv"));
    
    unsigned int set_n = 1;
    while (csvrp.get_next(ps))
    {
        cout << "loaded param_set " << set_n++ << endl;
        
        cout << "no seg fault"<<endl;
        
        //setting up simulation
        RATETYPE analysis_interval = .1;

        simulation s(m, ps,
            arg_parse::get<int>("c", "cell-total", 200),
            arg_parse::get<int>("w", "total-width", 50),
            arg_parse::get<RATETYPE>("s", "step-size", 0.01),
            arg_parse::get<RATETYPE>("a", "analysis_interval", analysis_interval),
            arg_parse::get<RATETYPE>("t", "sim_time", 60) );
   
        DataLogger dl(&s,analysis_interval); 
        cout << "no seg fault"<<endl;
        s.initialize();
        OscillationAnalysis o(&dl,4,ph1);

        cout << "no seg fault"<<endl;
        //run simulation
        s.simulate();
        //s.print_delay()	
        o.testQueue();

        dl.exportDataToFile("../models/her_model_2014/data_out.csv");
    }
}
