#ifndef IO_CSVR_PARAM_HPP
#define IO_CSVR_PARAM_HPP

#include "csvr.hpp"
#include "core/param_set.hpp"


class csvr_param : public csvr
{
public:
    csvr_param();
    csvr_param(std::string const& pcfFileName);

    /**
     *  Get/Return Next Set
     *
     *  usage
     *      For loading the next set of parameters in the file stream to the _critical_values, _delay_sets, and _rates_base of the param_set pLoadTo
     *
     *  parameters
     *      pfLoadTo - The instance of param_set to load the data to. Remember that these file loading functions are static!
     *
     *  returns
     *      true - if successfully loaded the next set in ifstream
     *      false - if unsuccessful
     *
     *  notes
     *      If no sets exist and/or the end of the file has been reached, will return false
     *      Secondary version of function instead returns a copy of a param_set with the data loaded onto it
    */
    bool get_next(param_set& pfLoadTo);
    param_set get_next();

};

#endif
