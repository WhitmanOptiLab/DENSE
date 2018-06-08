#ifndef ANLYS_BASE_HPP
#define ANLYS_BASE_HPP

#include "core/observable.hpp"
#include "core/specie.hpp"
#include "io/csvw.hpp"

#include <memory>

/// Superclass for Analysis Objects
/// - observes passed "Observable"
/// - does not implement any analysis
class Analysis : public Observer {

  public:

    Analysis (
      Observable & observable,
      specie_vec const& species_vector,
      csvw * csv_out,
      unsigned min_cell, unsigned max_cell,
      Real start_time, Real end_time
    );

    virtual void show () {};

    virtual void update(ContextBase& start) = 0;

    virtual void finalize() = 0;

    void when_updated_by(Observable &) override;

    void when_unsubscribed_from(Observable &) override;

  protected:

    Real start_time, end_time;

    unsigned min, max;

    unsigned time = 0;

    specie_vec const observed_species_;

    std::unique_ptr<csvw> csv_out;

};

#endif
