#include "basic.hpp"

#include <limits>
#include <algorithm>
#include <iostream>

BasicAnalysis::BasicAnalysis (
  Observable * log,
  specie_vec const& species_vector,
  csvw * csv_writer,
  unsigned min_cell, unsigned max_cell,
  Real start_time, Real end_time
) :
  Analysis(*log, species_vector, csv_writer, min_cell, max_cell, start_time, end_time),
  mins(observed_species_.size(), std::numeric_limits<Real>::infinity()),
  maxs(observed_species_.size(), Real{0}),
  means(observed_species_.size(), Real{0}),
  mins_by_context(max - min, mins),
  maxs_by_context(max - min, maxs),
  means_by_context(max - min, means) {
};

void BasicAnalysis::update (ContextBase & begin) {
  for (unsigned cell_no = min; cell_no < max; ++cell_no) {
    for (std::size_t i = 0; i < observed_species_.size(); ++i) {
  		Real concentration = begin.getCon(observed_species_[i]);
      mins[i] = std::min(concentration, mins[i]);
      maxs[i] = std::max(concentration, maxs[i]);
      means[i] += concentration;
      mins_by_context[cell_no][i] = std::min(concentration, mins_by_context[cell_no][i]);
      maxs_by_context[cell_no][i] = std::max(concentration, maxs_by_context[cell_no][i]);
  		means_by_context[cell_no][i] += concentration;
  	}
    begin.advance();
  }
  ++time;
};

void BasicAnalysis::finalize () {
  for (auto & mean : means) {
    mean /= time * (max - min);
  }
  for (auto & cell_means : means_by_context) {
    for (auto & mean : cell_means) {
      mean /= time;
    }
  }
  show();
};

void BasicAnalysis::show () {
  if (csv_out) {
    *csv_out << "\n\ncells " << min << '-' << max << ',';
    for (specie_id species : observed_species_) {
      *csv_out << specie_str[species] << ",";
    }

    csv_out->add_div("\nmin,");
    for (auto min : mins) {
      csv_out->add_data(min);
    }

    csv_out->add_div("\navg,");
    for (auto mean : means) {
      csv_out->add_data(mean);
    }

    csv_out->add_div("\nmax,");
    for (auto max : maxs) {
      csv_out->add_data(max);
    }
  } else {
    for (unsigned i = min; i < max; ++i) {
      std::cout << "Cell " << i << " (min, avg, max)\n";
      for (std::size_t s = 0; s < observed_species_.size(); ++s) {
        std::cout << specie_str[observed_species_[s]] << ": (" << mins[s] << ", " << means[s] << ", " << maxs[s] << ")\n";
      }
      std::cout << '\n';
    }
  }
};
