#include "basic.hpp"

using namespace std;
#include <limits>
#include <algorithm>
#include <iostream>

BasicAnalysis::BasicAnalysis (
  Observable * log,
  specie_vec const& species_vector,
  csvw * csv_writer,
  int min, int max,
  Real start_time, Real end_time
) :
  Analysis(log, species_vector, csv_writer, min, max, start_time, end_time),
  mins(ucSpecieOption.size(), std::numeric_limits<Real>::infinity()),
  maxs(ucSpecieOption.size(), Real{0}),
  means(ucSpecieOption.size(), Real{0}),
  mins_by_context(max - min, mins),
  maxs_by_context(max - min, maxs),
  means_by_context(max - min, means) {
};

void BasicAnalysis::update (ContextBase & begin) {
  for (unsigned cell_no = min; cell_no < max; ++cell_no) {
    for (std::size_t i = 0; i < ucSpecieOption.size(); ++i) {
  		Real concentration = begin.getCon(ucSpecieOption[i]);
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


void BasicAnalysis :: finalize(){
    // for each cell from max to min
    for (int c=0; c<max-min; c++){
        for (int s=0; s<ucSpecieOption.size(); s++){
            if (c==0){
                averages[s] = (averages[s]/time)/(max-min);
            }
            avgs_by_context[c][s] = avgs_by_context[c][s] / time;
        }
    }

    // only output min avg and max for now
    if (unFileOut)
    {
        // column label setup
        unFileOut->add_div("\n\ncells "+to_string(min)+"-"+to_string(max)+",");
        for (const specie_id& lcfID : ucSpecieOption)
            unFileOut->add_div(specie_str[lcfID] + ",");
        
        // row label and data
        unFileOut->add_div("\nmin,");
        for (int s=0; s<ucSpecieOption.size(); s++)
            unFileOut->add_data(mins[s]);
        
        unFileOut->add_div("\navg,");
        for (int s=0; s<ucSpecieOption.size(); s++)
            unFileOut->add_data(averages[s]);

        unFileOut->add_div("\nmax,");
        for (int s=0; s<ucSpecieOption.size(); s++)
            unFileOut->add_data(maxs[s]);
    } else {
        for (int i = min; i < max; i++) {
            std::cout << "Cell " << i << "(min, avg, max)" << std::endl;
            for (int s = 0; s < ucSpecieOption.size(); s++) {
               std::cout << specie_str[ucSpecieOption[s]] << ": (" << mins[s] << ',' << averages[s] << ',' << maxs[s] << ')' << std::endl;
            }
            std::cout << std::endl;
        }
    }
}
