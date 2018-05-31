#ifndef ANLYS_BASE_HPP
#define ANLYS_BASE_HPP

#include "core/observable.hpp"
#include "core/specie.hpp"
#include "io/csvw.hpp"

#include <memory>

/*
* Superclass for Analysis Objects
* - observes passed "Observable"
* - does not implement any analysis
*/
class Analysis : public Observer {

  public:

    Analysis (
      Observable * log,
      specie_vec const& species_vector,
      csvw * csv_out,
      int min, int max,
      Real start_time, Real end_time
    );

    virtual void show () {};

  protected:

    int time;

    specie_vec const ucSpecieOption;

    std::unique_ptr<csvw> csv_out;

};

#endif
