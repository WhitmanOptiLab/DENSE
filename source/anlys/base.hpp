#ifndef ANLYS_BASE_HPP
#define ANLYS_BASE_HPP

#include "core/observable.hpp"
#include "core/specie.hpp"
#include "io/csvw.hpp"

#include <memory>


/// Observer that restricts updates to a specific range of cells and times
/// Used as a stepping-stone to refactoring the Observer/Observable interface
class PickyObserver : public Observer {

  public:

    PickyObserver(Observable & observable, int min, int max, RATETYPE start_time, RATETYPE end_time);

    void when_updated_by(Observable &) override;

    void when_unsubscribed_from(Observable &) override;

    virtual void update(ContextBase& start) = 0;

    virtual void finalize() = 0;

  protected:

    Real start_time, end_time;

    int min, max;

};


/// Superclass for Analysis Objects
/// - observes passed "Observable"
/// - does not implement any analysis
class Analysis : public PickyObserver {

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
