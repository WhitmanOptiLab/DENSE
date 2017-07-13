#ifndef IO_CSVR_SIM_HPP
#define IO_CSVR_SIM_HPP

#include "csvr.hpp"
#include "core/context.hpp"
#include "core/observable.hpp"
#include "core/specie.hpp"

#include <map>
#include <vector>


class csvr_sim : public csvr, public Observable
{
public:
    class sim_ct : public ContextBase
    {
        friend class csvr_sim;
    public:
        sim_ct();

        CPUGPU_FUNC
        virtual RATETYPE getCon(specie_id sp) const;
        CPUGPU_FUNC
        virtual void advance();
        CPUGPU_FUNC
        virtual bool isValid() const;
        CPUGPU_FUNC
        virtual void set(int c);
    
    private:
        std::vector< std::map<specie_id, RATETYPE> > iRate;
        int iIter;
    };



    csvr_sim(const std::string& pcfFileName, const specie_vec& pcfSpecieVec);
    virtual ~csvr_sim();

    void run();

private:
    const specie_vec icSpecieVec;
    unsigned int iCellTotal;

};

#endif
