#ifndef IO_CSVR_SIM_HPP
#define IO_CSVR_SIM_HPP

#include "csvr.hpp"
#include "sim/base.hpp"
#include "core/observable.hpp"
#include "core/specie.hpp"

#include <map>
#include <vector>


class csvr_sim : public csvr, public Observable
{
public:
    class sim_ct : public dense::Context
    {
        friend class csvr_sim;
    public:
        sim_ct();

        IF_CUDA(__host__ __device__)
        virtual Real getCon(specie_id sp) const;
        IF_CUDA(__host__ __device__)
        virtual void advance();
        IF_CUDA(__host__ __device__)
        virtual bool isValid() const;
        IF_CUDA(__host__ __device__)
        virtual void set(int c);

    private:
        std::vector<std::map<specie_id, Real>> iRate;
        unsigned iIter;
    };



    csvr_sim(std::string const& pcfFileName, specie_vec const& pcfSpecieVec);
    virtual ~csvr_sim();

    int getCellTotal();
    Real getAnlysIntvl();
    Real getTimeStart();
    Real getTimeEnd();
    int getCellStart();
    int getCellEnd();

    void run() final;

private:
    // Required for csvr_sim
    specie_vec iSpecieVec;
    unsigned iCellTotal;
    bool iTimeCol;

    // For everyone else to get()
    Real iAnlysIntvl, iTimeStart, iTimeEnd;
    int iCellStart, iCellEnd;
};

#endif
