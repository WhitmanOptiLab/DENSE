#include <limits>
#include <algorithm>
#include <iostream>
#include <string>
#include <stdexcept>
#include <cmath>

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
  variance(observed_species.size(), Real{0}),
  mins_by_context(Analysis<>::max - Analysis<>::min, mins),
  maxs_by_context(Analysis<>::max - Analysis<>::min, maxs),
  means_by_context(Analysis<>::max - Analysis<>::min, means) {
	finalized = false;
}

template <typename Simulation>
void BasicAnalysis<Simulation>::update (Simulation& simulation, std::ostream&) {
  Natural n_min1 = Analysis<>::samples;
  for (std::size_t i = 0; i < this->observed_species_.size(); ++i) {
    Real concsum = 0.0;
    for (Natural cell_no = this->min; cell_no < this->max; ++cell_no) {
  		Real concentration = simulation.get_concentration(cell_no, this->observed_species_[i]);
      mins[i] = std::min(concentration, mins[i]);
      maxs[i] = std::max(concentration, maxs[i]);
      mins_by_context[cell_no][i] = std::min(concentration, mins_by_context[cell_no][i]);
      maxs_by_context[cell_no][i] = std::max(concentration, maxs_by_context[cell_no][i]);
      means_by_context[cell_no][i] = 
        (means_by_context[cell_no][i]*n_min1 + concentration)/(n_min1 + 1);
      concsum += concentration;
  	}

    Real oldmean = means[i];
    means[i] = (means[i]*n_min1 + (concsum/(this->max - this->min)))/(n_min1 + 1);
    Real meandiff = oldmean - means[i];
    Real sumsqdiff = 0.0f;
    for (Natural cell_no = this->min; cell_no < this->max; ++cell_no) {
      Real concdiff = simulation.get_concentration(cell_no, this->observed_species_[i]) - means[i];
      sumsqdiff += concdiff*concdiff;
    }
    Real prev_samples = n_min1*(this->max - this->min);
    variance[i] = ((prev_samples - 1)*variance[i] + prev_samples*meandiff*meandiff + sumsqdiff) 
                                    / (prev_samples + (this->max - this->min) - 1 );
  }
  ++Analysis<>::samples;
}

template <typename Simulation>
void BasicAnalysis<Simulation>::finalize () {

  if(!finalized){
    finalized = true;
  }
	detail.concs = means;
	detail.other_details.emplace_back(mins);
	detail.other_details.emplace_back(maxs);
}

template<typename Simulation>
Details BasicAnalysis<Simulation>::get_details(){
	return detail;
}

#include <iomanip>

template <typename Simulation>
void BasicAnalysis<Simulation>::show (csvw * csv_out) {
  Analysis<>::show(csv_out);
  if (csv_out) {
    auto & out = *csv_out;
    //out << "Species,Minimum Concentration,Mean Concentration,Maximum Concentration,Standard Deviation \n";
    //out << std::scientific << std::setprecision(5);
    for (specie_id species : Analysis<>::observed_species_) {
      out << specie_str[species] << "\n" <<
        "Minimum Concentration:\n" << mins[species] << "\n" <<
        "Mean Concentration:\n" << means[species] << "\n" <<
        "Maximum Concentration:\n" << maxs[species] << "\n" <<
        "Standard Deviation:\n" << std::sqrt(variance[species]) << "\n" ;
    }
  }
}
