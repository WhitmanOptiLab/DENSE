#include "csvr.hpp"
#include "context.hpp"
#include "observable.hpp"
#include "simulation.hpp"
#include "specie.hpp"


class csvr_sim : public csvr, public Observable
{
public:
    csvr_sim(const std::string& pcfFileName);
    virtual ~csvr_sim();

    void run();

private:
    class mini_ct : public ContextBase
    {
    public:
        mini_ct();

        CPUGPU_FUNC
        virtual RATETYPE getCon(specie_id sp) const;
        CPUGPU_FUNC
        virtual void advance();
        CPUGPU_FUNC
        virtual bool isValid() const;
    
    private:
        RATETYPE iRate[NUM_SPECIES];
        specie_id iIter;
    }
};
