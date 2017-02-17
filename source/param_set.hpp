#ifndef PARAM_SET_HPP
#define PARAM_SET_HPP

#include <utility>

#include "reaction.hpp"

using namespace std;

#define NUM_SECTIONS 2
#define MAX_CONDS_ANY 2

class param_set{
private:
   
    
    int index; // The index, i.e. how many mutants run before this one
    char* print_name; // The mutant's name for printing output
    char* dir_name; // The mutant's name for making its directory
    int num_knockouts; // The number of knockouts required to achieve this mutant
    int knockouts[2]; // The indices of the concentrations to knockout (num_knockouts determines how many indices to knockout)
    double overexpression_rate; // Which gene should be overexpressed, -1 = none
    double overexpression_factor; // Overexpression factors with 1=100% overexpressed, if 0 then no overexpression
    int induction; // The induction point for mutants that are time sensitive
    int recovery;
    //con_levels cl; // The concentration levels at the end of this mutant's posterior simulation run
    /*
     bool initialized; // Whether or not this struct's data have been initialized
     int num_con_levels; // The number of concentration levels this struct stores (not necessarily the total number of concentration levels)
     int time_steps; // The number of time steps this struct stores concentrations for
     int cells; // The number of cells this struct stores concentrations for
     concentration_level<double> cons; // A three dimensional array that stores [concentration levels][time steps][cells] in that order
     int* active_start_record; // Record of the start of the active PSM at each time step
     int* _active_start_record; // Record of the start of the active PSM at each time step
     int* active_end_record; // Record of the end of the active PSM at each time step
     int* _active_end_record; // Record of the end of the active PSM at each time step
     */
    //double (*tests[2])(mutant_data&, features&); // The posterior and anterior conditions tests
    //int (*wave_test)(pair<int, int>[], int, mutant_data&, int, int); // The traveling wave conditions test
    int num_conditions[NUM_SECTIONS+1]; // The number of conditions this mutant is tested on
    double cond_scores[NUM_SECTIONS+1][MAX_CONDS_ANY]; // The score this mutant can achieve for each condition
    double max_cond_scores[NUM_SECTIONS]; // The maximum score this mutant can achieve for each section
    bool secs_passed[NUM_SECTIONS]; // Whether or not this mutant has passed each section's conditions
    double conds_passed[NUM_SECTIONS][1 + MAX_CONDS_ANY]; // The score this mutant achieved for each condition when run
    //feature feat; // The oscillation features this mutant produced when run
    int print_con; // The index of the concentration that should be printed (usually mh1)
    bool only_post; //indicating if only the posterior is simulated
    RATETYPE* _sets;
    double rates_base[NUM_REACTIONS]; // Base rates taken from the current parameter set
};


#endif

