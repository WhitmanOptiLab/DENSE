#ifndef PARAM_SET_HPP
#define PARAM_SET_HPP

#include <fstream>
#include <string>
#include <utility>

#include "reaction.hpp"

#define NUM_SECTIONS 2
#define MAX_CONDS_ANY 2

class param_set{
public:
   
    
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
    
    // currently in use
    RATETYPE _critical_values[NUM_CRITICAL_SPECIES];
    RATETYPE _delay_sets[NUM_DELAY_REACTIONS];
    RATETYPE _rates_base[NUM_REACTIONS];
    
    void printall() const
    {
        for (unsigned int i=0; i<NUM_CRITICAL_SPECIES; i++)
            printf("%f, ", _critical_values[i]);
        printf("\n");
        for (unsigned int i=0; i<NUM_DELAY_REACTIONS; i++)
            printf("%f, ", _delay_sets[i]);
        printf("\n");
        for (unsigned int i=0; i<NUM_REACTIONS; i++)
            printf("%f, ", _rates_base[i]);
        printf("\n");
    }
    
    
    
    /**
     *  Open Input File Stream
     *
     *  usage
     *      For choosing a file to load param_set fields from
     *  
     *  parameters
     *      pFileName - file name to load from
     *
     *  returns
     *      true - if successfully loaded file specified by pFileName
     *      false - if unsuccessful
     *  
     *  notes
     *      If current_ifstream is already set to something, close_ifstream() will automatically be called.
    */
    static bool open_ifstream(const std::string& pFileName);
    
    /**
     *  Close Input File Stream
     *
     *  usage
     *      For closing current_ifstream
     *      Does not need to be called before open_ifstream(); see open_ifstream() documentation for more info
     *      Instead, you will only probably use this where a param_set destructor would go
     *
     *  notes
     *      Will never cause an error, even if current_ifstream is uninitialized
    */
    static void close_ifstream();
    
    /**
     *  Get Total/Remaining Counts of Data Sets
     *
     *  usage
     *      For getting the total/remaining number of sets of data in the opened ifstream
     *      For total, does not matter if you have already started using load_next_set()
     *
     *  returns
     *      Total/Remaining number of data sets in current_ifstream
     *      Returns 0 if no file has been loaded or if there is indeed no data in current_ifstream
    */
    static unsigned int get_set_total();
    static unsigned int get_set_remaining();
    
    /**
     *  Load Next Set
     *
     *  usage
     *      For loading the next set of parameters in the file stream to the _critical_values, _delay_sets, and _rates_base of the param_set pLoadTo
     *
     *  parameters
     *      pLoadTo - The instance of param_set to load the data to. Remember that these file loading functions are static!
     *
     *  returns
     *      true - if successfully loaded the next set in current_ifstream
     *      false - if unsuccessful
     *
     *  notes
     *      If no sets exist and/or the end of the file has been reached, will return false
     *      Secondary version of function instead returns a copy of a param_set with the data loaded onto it
    */
    static bool load_next_set(param_set &pLoadTo);
    static param_set load_next_set();
    
private:
    // Current input file stream for loading in param_set fields
    static std::ifstream current_ifstream;
    
    // Counters for total and remaining data sets in current_ifstream
    static unsigned int current_total;
    static unsigned int current_remaining;
};


#endif

