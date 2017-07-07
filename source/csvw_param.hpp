#ifndef CSVW_PARAM_HPP
#define CSVW_PARAM_HPP

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
    csvw_param(const std::string& pcfFileName, const param_type& pcfType);
    virtual ~csvw_param();
};

#endif
