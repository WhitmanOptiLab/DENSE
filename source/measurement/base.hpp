#ifndef ANLYS_BASE_HPP
#define ANLYS_BASE_HPP

#include "core/observable.hpp"
#include "core/specie.hpp"
#include "io/csvw.hpp"
#include "sim/base.hpp"

#include <memory>
using dense::Simulation;

/// Superclass for Analysis Objects
/// - observes passed "Observable"
/// - does not implement any analysis
class Analysis : public Observer {

  public:

    Analysis (
      specie_vec const& species_vector,
      unsigned min_cell, unsigned max_cell,
      Real start_time, Real end_time
    );

    virtual void show (csvw * = nullptr) {};

    virtual void update(dense::Context<> start) = 0;

    virtual void finalize() = 0;

    void when_updated_by(Observable &) override;

    void when_unsubscribed_from(Observable &) override;

  protected:

    specie_vec const observed_species_;

    Real start_time, end_time;

    unsigned const min, max;

    unsigned time = 0;

};

class Measurement {

  using Collector = Analysis;

};
/*
metrics have to be associated with parameter sets that came in

there needs to be a list of metrics eventually, in standard format across the parameter_set s, aka a single score you can compare

so metrics need to be the same across parameter sets

you have to take the same measurements across all the simulations in order to compare them.
which means you can\'t have measurements that use more than one simulation, because then simulations can\'t be judged in their own right.

a metric is a methodology for judging a given experiment

but there can be multiple measurements per simulation
measurements can only be taken on a single simulation

-> simulation -> measurement -+
                              |
-> simulation -> measurement -+->

very beginning:
- you have a group of measurements you want to take for each simulation
- you have a metric you can apply

- a metric describes what measurements to take and how to assess the results


you start out with a population of Parameter_Sets


*/



#endif
