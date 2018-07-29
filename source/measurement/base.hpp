#ifndef ANLYS_BASE_HPP
#define ANLYS_BASE_HPP

#include "core/specie.hpp"
#include "io/csvw.hpp"
#include "sim/base.hpp"

#include <memory>
#include <limits>

using dense::Simulation;

/// Superclass for Analysis Objects
template <typename Simulation = void>
class Analysis;

template <>
class Analysis<void> {

  public:

    Analysis (
      std::vector<Species> const& species_vector,
      std::pair<dense::Natural, dense::Natural> cell_range,
      std::pair<Real, Real> time_range = { 0, std::numeric_limits<Real>::infinity() }
    ):
      observed_species_{species_vector},
      start_time{time_range.first},
      end_time{time_range.second},
      min{cell_range.first},
      max{cell_range.second} {
    }

    virtual void show (csvw* csv_out = nullptr) {
      if (csv_out) {
        auto & out = *csv_out;
        out << "# Showing cells " << min << '-' << max;
        if (start_time != 0.0) {
          out << " between " << start_time << " min and ";
        } else {
          out << " until ";
        }
        out << time << " min\n";
      }
    }

  protected:

    std::vector<Species> observed_species_;

    Real start_time, end_time;

    dense::Natural min, max;

    Real time = 0;

    dense::Natural samples = 0;

};

template <typename Simulation>
class Analysis : public Analysis<> {

  static_assert(std::is_base_of<dense::Simulation, Simulation>::value,
    "requires DerivedFrom<dense::Simulation, Simulation>");

  public:

    using Analysis<>::Analysis;

    virtual void update(Simulation& start, std::ostream& log = std::cout) = 0;

    virtual Analysis* clone() const = 0;

    virtual void finalize() = 0;

    void when_updated_by(Simulation & simulation, std::ostream& log);

};

#include "base.ipp"

#endif
