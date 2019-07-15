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
  sd(observed_species.size(), Real{0}),
  mins_by_context(Analysis<>::max - Analysis<>::min, mins),
  maxs_by_context(Analysis<>::max - Analysis<>::min, maxs),
  means_by_context(Analysis<>::max - Analysis<>::min, means) {
	finalized = false;
}

template <typename Simulation>
void BasicAnalysis<Simulation>::update (Simulation& simulation, std::ostream&) {
  Real mean = 0.0, M2 = 0.0;
  int n = 0;
  for (Natural cell_no = this->min; cell_no < this->max; ++cell_no) {
    for (std::size_t i = 0; i < this->observed_species_.size(); ++i) {
  		Real concentration = simulation.get_concentration(cell_no, this->observed_species_[i]);
      mins[i] = std::min(concentration, mins[i]);
      maxs[i] = std::max(concentration, maxs[i]);
      means[i] += concentration;
      mins_by_context[cell_no][i] = std::min(concentration, mins_by_context[cell_no][i]);
      maxs_by_context[cell_no][i] = std::max(concentration, maxs_by_context[cell_no][i]);
  		means_by_context[cell_no][i] += concentration;
      Real delta = concentration - mean;
      mean += delta / (n + 1);
      M2 += delta * (concentration - mean);
      sd[i] = sqrt(M2 / (n+1));
      n++;
  	}
  }
  ++Analysis<>::samples;
}

template <typename Simulation>
void BasicAnalysis<Simulation>::finalize () {

if(!finalized){
	finalized = true;
}
  for (auto & cell_means : means_by_context) {
    for (auto & mean : cell_means) {
      mean /= Analysis<>::samples;
    }
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
    out << "Species,Minimum Concentration,Mean Concentration,Maximum Concentration,Standard Deviation \n";
    //out << std::scientific << std::setprecision(5);
    for (specie_id species : Analysis<>::observed_species_) {
      out << specie_str[species] << "," <<
        mins[species] << "," <<
        means[species] / (Analysis<>::samples * (Analysis<>::max - Analysis<>::min)) << "," <<
        maxs[species] << "," <<
        sd[species] << "\n" ;
    const char* path = "./test/ndiff/test.out";
    std::ofstream outfile (path);
    outfile<<specie_str[species]<<"\n"<<
    means[species] / (Analysis<>::samples * (Analysis<>::max - Analysis<>::min))<<"\n"<<
    sd[species]<<std::endl;   
    outfile.close(); 
    }
  }
}