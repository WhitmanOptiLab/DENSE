#include "arg_parse.hpp"
#include "simulation_determ.hpp"
#include "simulation_stoch.hpp"
#include "model_impl.hpp"
#include "context_determ.hpp"
#include "analysis.hpp"
#include "csvr_param.hpp"
#include <iostream>
#include <chrono>
#include "csvw_sim.hpp"

int main(int argc, char *argv[]) {
    arg_parse::init(argc, argv);
    
    int total_width = arg_parse::get<int>("w", "total-width", 5);

    //setting up model
    model m(arg_parse::get<string>("g", "gradients", ""),
        arg_parse::get<string>("v", "perturb", ""), total_width);
    
    //setting up param_set
    param_set ps;

    csvr_param csvrp(arg_parse::get<string>("p", "param-sets", "../models/her_model_2014/param_sets.csv")); // MAKE SURE THIS IS RIGHT!!!
    
    if (csvrp.is_open())
    {
        unsigned int set_n = 0;
        while (csvrp.get_next(ps))
        {
            cout << "loaded param_set " << set_n++ << endl;
            
            //setting up simulation
            RATETYPE analysis_interval = arg_parse::get<RATETYPE>("a","analysis_interval",0.01);
   
           simulation_base* s; 
           if (arg_parse::get<bool>("d", "determ", false))
           { 
             s = new simulation_determ(m, ps,
                arg_parse::get<int>("c", "cell-total", 10),
                total_width,
                arg_parse::get<RATETYPE>("s", "step-size", 0.01),
                analysis_interval,
                arg_parse::get<RATETYPE>("t", "sim_time", 6) );
           }
           else
           {
             s = new simulation_stoch(m, ps,
                arg_parse::get<int>("c", "cell-total", 10),
                arg_parse::get<int>("w", "total-width", 5),
                analysis_interval,
                arg_parse::get<RATETYPE>("t", "sim_time", 6),
                arg_parse::get<int>("r","seed",chrono::system_clock::now().time_since_epoch().count())
             );

           }
//        simulation_stoch s(m, ps,1,1,0.1,6);

           s->initialize();
          specie_vec specie_option;
         arg_parse::get<specie_vec>("o", "specie-option", &specie_option, false);
         csvw_sim write("outfile",0.01,200,specie_option,s);
           OscillationAnalysis o(s,analysis_interval,arg_parse::get<RATETYPE>("r","local_range",4),mh1);
            OscillationAnalysis o1(s,analysis_interval,arg_parse::get<RATETYPE>("r","local_range",4),mh7);
            OscillationAnalysis o2(s,analysis_interval,arg_parse::get<RATETYPE>("r","local_range",4),md);
           //            BasicAnalysis a(s);
            //run simulation
           s->simulate();
            //s.print_delay()	
          //  o1.test();
          //  o2.test();
          // o.test();
//            a.test();
        }
    }
}
