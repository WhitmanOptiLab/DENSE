#ifndef CSVW_SIM_H
#define CSVW_SIM_H

#include "csvw.hpp"
#include "context.hpp"
#include "observable.hpp"


class csvw_sim : public csvw, public Observer
{
public:
    csvw_sim(const std::string& pcfFileName);
    virtual ~csvw_sim();

    virtual void update(ContextBase& pfStart);
};

#endif
