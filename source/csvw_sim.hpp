#ifndef CSVW_SIM_HPP
#define CSVW_SIM_HPP

#include "csvw.hpp"
#include "context.hpp"
#include "observable.hpp"


class csvw_sim : public csvw, public Observer
{
public:
    csvw_sim(const std::string& pcfFileName, Observable *pnObl);
    virtual ~csvw_sim();
    
    virtual void finalize(ContextBase& start);
    virtual void update(ContextBase& pfStart);
};

#endif
