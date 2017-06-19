#include "csvr.hpp"
#include "context.hpp"
#include "observable.hpp"
#include "specie.hpp"


class csvr_sim : public csvr, public Observable
{
public:
    class mini_ct : public ContextBase
    {
        friend class csvr_sim;
    public:
        mini_ct();

        CPUGPU_FUNC
        virtual RATETYPE getCon(specie_id sp) const;
        CPUGPU_FUNC
        virtual void advance();
        CPUGPU_FUNC
        virtual bool isValid() const;
        CPUGPU_FUNC
        virtual void reset();
    
    private:
        RATETYPE iRate[NUM_SPECIES];
        int iIter;
    };



    csvr_sim(const std::string& pcfFileName);
    virtual ~csvr_sim();

    void run();
};
