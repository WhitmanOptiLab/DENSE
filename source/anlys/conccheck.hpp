#ifndef ANLYS_CONCCHECK_HPP
#define ANLYS_CONCCHECK_HPP

#include "base.hpp"

class ConcentrationCheck : public Analysis {

  private:
    RATETYPE lower_bound, upper_bound;
    specie_id target_specie;

  public:
    ConcentrationCheck(Observable *data_source, int min, int max, 
                        RATETYPE lowerB, RATETYPE upperB, RATETYPE startT, RATETYPE
                        endT, specie_id t_specie=(specie_id)-1) :
                        Analysis(data_source, min, max, startT, endT), target_specie(t_specie),
                        lower_bound(lowerB), upper_bound(upperB){}

    void update(ContextBase& start) {

        RATETYPE con;

        for (int c=min; c<max; c++){
            if (target_specie > -1){
                con = start.getCon(target_specie);
                if (con<lower_bound || con>upper_bound) {
                    subject->abort();
                }
            }else{
                for (int s=0; s<NUM_SPECIES; s++){
                    RATETYPE con = start.getCon((specie_id) s);
                    if (con<lower_bound || con>upper_bound) {
                        subject->abort();
                    }
                }
            }
        }
    }

    void finalize(){} 
};
#endif
