#ifndef CSVW_PARAM_HPP
#define CSVW_PARAM_HPP

#include "csvw.hpp"


class csvw_param : public csvw
{
public:
    csvw_param(const std::string& pcfFileName);
    virtual ~csvw_param();
};

#endif
