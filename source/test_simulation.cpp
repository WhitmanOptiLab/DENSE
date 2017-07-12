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
           int cells_total = arg_parse::get<int>("c","cell-total",10);
           RATETYPE sim_time = arg_parse::get<RATETYPE>("t","sim_time",60);
   
           simulation_base* s; 
           if (arg_parse::get<bool>("d", "determ", false))
           { 
             s = new simulation_determ(m, ps,
                cells_total,
                total_width,
                arg_parse::get<RATETYPE>("s", "step-size", 0.01),
                analysis_interval,
                sim_time);
           }
           else
           {
             s = new simulation_stoch(m, ps,
                cells_total,
                arg_parse::get<int>("w", "total-width", 5),
                analysis_interval,
                sim_time,
                arg_parse::get<int>("r","seed",chrono::system_clock::now().time_since_epoch().count())
             );

           }

           s->initialize();
           specie_vec specie_option;
           arg_parse::get<specie_vec>("o", "specie-option", &specie_option, false);
           csvw_sim write("outfile",0.01,200,specie_option,s,0,cells_total,20,60);
           OscillationAnalysis o(s,analysis_interval,arg_parse::get<RATETYPE>("lr","local_range",4),mh7,0,cells_total,30, 40);
          // OscillationAnalysis o1(s,analysis_interval,arg_parse::get<RATETYPE>("lr","local_range",4),mh7,0,cells_total,0,sim_time);
          // OscillationAnalysis o2(s,analysis_interval,arg_parse::get<RATETYPE>("lr","local_range",4),md,0, cells_total,0,sim_time);
           BasicAnalysis a(s,0,cells_total,10,15);
          // ConcentrationCheck cc(s,0,cells_total,-1,10,0,sim_time,ph1);
            //run simulation
           s->simulate();
            //s.print_delay()	
          // o1.test();
          // o2.test();
           o.test();
           a.test();
        }
    }
}
