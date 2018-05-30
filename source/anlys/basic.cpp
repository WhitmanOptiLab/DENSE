#include "basic.hpp"

using namespace std;

/*
 * UPDATE_AVERAGES
 * arg "start": context iterator to access newest conc level with
 * arg "c": the cell the context inhabits
 * updates average concentration levels across all cells and per cell
*/
void BasicAnalysis :: update_averages(const ContextBase& start, int c){
	for (int i=0; i<ucSpecieOption.size(); i++){
		specie_id sid = (specie_id) ucSpecieOption[i];
        averages[i] += start.getCon(sid);
		avgs_by_context[c][i] += start.getCon(sid);
	}
}
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

/*
 * UPDATE_MINMAX
 * arg "start": context iterator to access newest conc level with
 * arg "c": the cell the context inhabits
 * updates min and max conc levels across all cells and per cell
*/
void BasicAnalysis :: update_minmax(const ContextBase& start, int c){

	for (int i=0; i<ucSpecieOption.size(); i++){
		specie_id sid = static_cast<specie_id>(ucSpecieOption[i]);
		RATETYPE conc = start.getCon(sid);
		if (conc > maxs[i]){
			maxs[i] = conc;
		}
		if (conc < mins[i]){
			mins[i] = conc;
		}
		if (conc > maxs_by_context[c][i]){
			maxs_by_context[c][i] = conc;
		}
		if (conc < mins_by_context[c][i]){
			mins_by_context[c][i] = conc;
		}
	}
}

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
