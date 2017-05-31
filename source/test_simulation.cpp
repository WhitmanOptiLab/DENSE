#include "arg_parse.hpp"
#include "simulation.hpp"
#include "model_impl.hpp"
#include "context_impl.hpp"
#include <iostream>

int main(int argc, char *argv[]) {
    arg_parse::init(argc, argv);
    
    //setting up model
    model m(arg_parse::get<bool>("G", false, "gradients"),
        arg_parse::get<bool>("P", false, "perturb"));
    
    //setting up param_set
    param_set ps;
    
    if (param_set::open_ifstream(arg_parse::get<string>("p", "../models/her_model_2014/param_list.csv", "param-list")))
    {
        unsigned int set_n = 1;
        while (param_set::load_next_set(ps))
        {
            cout << "loaded param_set " << set_n++ << endl;
            
            cout << "no seg fault"<<endl;
            
            //setting up simulation
            simulation s(m, ps,
                arg_parse::get<int>("c", 200, "cell-total"),
                arg_parse::get<int>("w", 50, "total-width"),
                arg_parse::get<RATETYPE>("s", 0.01, "step-size") );
            cout << "no seg fault"<<endl;
            s.initialize();
            cout << "no seg fault"<<endl;
            //run simulation
            s.simulate(arg_parse::get<int>("t", 60, "time"));
            //s.print_delay();
        }
    }
    
    ps.close_ifstream();
}
