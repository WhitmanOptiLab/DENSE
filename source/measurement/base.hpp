#ifndef ANLYS_BASE_HPP
#define ANLYS_BASE_HPP

#include "core/observable.hpp"
#include "core/specie.hpp"
#include "io/csvw.hpp"
#include "sim/base.hpp"

#include <memory>
#include <limits>

using dense::Simulation;

/// Superclass for Analysis Objects
/// - observes passed "Observable"
/// - does not implement any analysis
class Analysis : public Observer<Simulation> {

  public:

    Analysis (
      specie_vec const& species_vector,
      unsigned min_cell, unsigned max_cell,
      Real start_time = 0, Real end_time = std::numeric_limits<Real>::infinity()
    );

    virtual void show (csvw * = nullptr) {};

    virtual void update(dense::Context<> start) = 0;

    virtual void finalize() = 0;

    void when_updated_by(Simulation &) override;

    void when_unsubscribed_from(Simulation &) override;

  protected:

    specie_vec const observed_species_;

    Real start_time, end_time;

    unsigned const min, max;

    dense::Natural time = 0;

};

#endif
