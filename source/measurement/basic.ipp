#include <limits>
#include <algorithm>
#include <iostream>

template <typename Simulation>
BasicAnalysis<Simulation>::BasicAnalysis (
  std::vector<Species> const& observed_species,
  std::pair<dense::Natural, dense::Natural> cell_range,
  std::pair<Real, Real> time_range
) :
  Analysis<Simulation>(observed_species, cell_range, time_range),
  mins(observed_species.size(), std::numeric_limits<Real>::infinity()),
  maxs(observed_species.size(), Real{0}),
  means(observed_species.size(), Real{0}),
  mins_by_context(Analysis<>::max - Analysis<>::min, mins),
  maxs_by_context(Analysis<>::max - Analysis<>::min, maxs),
  means_by_context(Analysis<>::max - Analysis<>::min, means) {
}

template <typename Simulation>
void BasicAnalysis<Simulation>::update (Simulation& simulation, std::ostream&) {
  for (Natural cell_no = this->min; cell_no < this->max; ++cell_no) {
    for (std::size_t i = 0; i < this->observed_species_.size(); ++i) {
  		Real concentration = simulation.get_concentration(cell_no, this->observed_species_[i]);
      mins[i] = std::min(concentration, mins[i]);
      maxs[i] = std::max(concentration, maxs[i]);
      means[i] += concentration;
      mins_by_context[cell_no][i] = std::min(concentration, mins_by_context[cell_no][i]);
      maxs_by_context[cell_no][i] = std::max(concentration, maxs_by_context[cell_no][i]);
  		means_by_context[cell_no][i] += concentration;
  	}
  }
  ++Analysis<>::samples;
}

template <typename Simulation>
void BasicAnalysis<Simulation>::finalize () {
  for (auto & cell_means : means_by_context) {
    for (auto & mean : cell_means) {
      mean /= Analysis<>::samples;
    }
  }
}

#include <iomanip>

template <typename Simulation>
void BasicAnalysis<Simulation>::show (csvw * csv_out) {
  Analysis<>::show(csv_out);
  if (csv_out) {
    auto & out = *csv_out;
    out << "Species,Minimum Concentration,Mean Concentration,Maximum Concentration\n";
    //out << std::scientific << std::setprecision(5);
    for (specie_id species : Analysis<>::observed_species_) {
      out << specie_str[species] << "," <<
        mins[species] << "," <<
        means[species] / (Analysis<>::samples * (Analysis<>::max - Analysis<>::min)) << "," <<
        maxs[species] << "\n";
    }
  }
}
