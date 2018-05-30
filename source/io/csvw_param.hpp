#ifndef IO_CSVW_PARAM_HPP
#define IO_CSVW_PARAM_HPP

#include "csvw.hpp"


enum param_type
{
    SETS,
    PERT,
    GRAD
};


class csvw_param : public csvw
{
public:
    csvw_param(std::string const& pcfFileName, param_type const& pcfType);
    virtual ~csvw_param();
};

#endif
