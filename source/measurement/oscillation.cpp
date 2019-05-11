#include "oscillation.hpp"
#include "sim/base.hpp"

/*
 * FINALIZE
 * finds peaks and troughs in final half-window of simulation data
 * precondition: the simulation has finished
*/
void OscillationAnalysis :: finalize(){
    int timeTemp = time;
    for (std::size_t s = 0; s < observed_species_.size(); ++s)
    {
        for (unsigned c = 0; c < max-min; ++c){
            time = timeTemp;
            while (windows[s][c].getSize()>=(range_steps/2)&&bst[s][c].size()>0){
                Real removed = windows[s][c].dequeue();
                bst[s][c].erase(bst[s][c].find(removed));
                //std::cout<<"bst size="<<bst[s][c].size()<<'\n';
                checkCritPoint(s, c);
                ++time;
            }
            calcAmpsAndPers(s, c);
        }
    }
    show();
}

void OscillationAnalysis::show () {
  if (csv_out)
  {
      for (unsigned c = min; c < max; ++c)
      {
          std::vector<Real> avg_peak(observed_species_.size());
          for (std::size_t s = 0; s < observed_species_.size(); ++s)
          {
              Real peak_count = 0;
              avg_peak[s] = 0.0;

              for (std::size_t pt = 0; pt < peaksAndTroughs[s][c].size(); ++pt)
              {
                  crit_point cp = peaksAndTroughs[s][c][pt];
                  if (cp.is_peak)
                  {
                      avg_peak[s] += cp.conc;
                      ++peak_count;
                  }
              }

              if (peak_count > 0) avg_peak[s] /= peak_count;
          }

          csv_out->add_div("\n\ncell " + std::to_string(c) + ",");
          for (specie_id const& lcfID : observed_species_)
              csv_out->add_div(specie_str[lcfID] + ",");

          csv_out->add_div("\navg peak,");
          for (std::size_t s = 0; s < observed_species_.size(); ++s)
              csv_out->add_data(avg_peak[s]);

          csv_out->add_div("\navg amp,");
          for (std::size_t s = 0; s < observed_species_.size(); ++s)
              csv_out->add_data(amplitudes[s][c]);

          csv_out->add_div("\navg per,");
          for (std::size_t s = 0; s < observed_species_.size(); ++s)
              csv_out->add_data(periods[s][c]);
      }
  }
}

/*
 * GET_PEAKS_AND_TROUGHS
 * advances the local range window and identifies critical points
 * arg "start": context iterator to access conc levels with
 * arg "c": the cell the context inhabits
*/
void OscillationAnalysis :: get_peaks_and_troughs(dense::Context const& start, int c){

    for (std::size_t i = 0; i < observed_species_.size(); ++i)
    {
        Real added = start.getCon(observed_species_[i]);
        windows[i][c].enqueue(added);
        bst[i][c].insert(added);
        if ( windows[i][c].getSize() == range_steps + 1) {
            Real removed = windows[i][c].dequeue();
            bst[i][c].erase(bst[i][c].find(removed));
        }
        if ( windows[i][c].getSize() < range_steps/2) {
            return;
        }
	    checkCritPoint(i, c);
    }
}

/*
 * CHECKCRITPOINT
 * determines if a particular specie conc level in a particular cell is a peak, trough, or neither
 * arg "c": the cell this concentration level is found in
*/
void OscillationAnalysis :: checkCritPoint(int s, int c){
	Real mid_conc = windows[s][c].getVal(windows[s][c].getCurrent());
	if ((mid_conc==*bst[s][c].rbegin())&&!(mid_conc==*bst[s][c].begin())){
		addCritPoint(s,c,true,std::max<Real>(0.0,(time-range_steps/2)*analysis_interval + start_time),mid_conc);
	}
	else if (mid_conc==*bst[s][c].begin()){
		addCritPoint(s,c,false,std::max<Real>(0.0,(time-(range_steps/2))*analysis_interval + start_time),mid_conc);
	}
}

/*
 * ADDCRITPOINT
 * adds the peak or trough to the "crit_point" vector if it is an oscillating feature (no two peaks or two troughs in a row)
 * arg "context": context iterator to access conc levels with
 * arg "isPeak": bool is true if the conc level is a peak and false if it is a trough
 * arg "minute": time, in minutes, that the critical point occurs
 * arg "concentration": the concentration level of the critical point
*/
void OscillationAnalysis :: addCritPoint(int s, int context, bool isPeak, Real minute, Real concentration){
	crit_point crit;
	crit.is_peak = isPeak;
	crit.time = minute;
	crit.conc = concentration;

	if (peaksAndTroughs[s][context].size() > 0){
		crit_point prev_crit = peaksAndTroughs[s][context].back();
		if (prev_crit.is_peak == crit.is_peak){
			if ((crit.is_peak && crit.conc >= prev_crit.conc)||(!crit.is_peak && crit.conc <= prev_crit.conc)){
				peaksAndTroughs[s][context].back() = crit;
			}
		}
		else {
			peaksAndTroughs[s][context].push_back(crit);
		}
	} else {
		peaksAndTroughs[s][context].push_back(crit);
	}
}

/*
 * UPDATE
 * called by attached observables in "notify" function
 * main analysis function
 * arg "start": context iterator to access con levels with
 * precondition: "start" inhabits cell 0
 * postcondtion: "start" inhabits an invalid cell
*/
void OscillationAnalysis::update (dense::Context & start) {
	for (unsigned c = min; c < max && start.isValid(); ++c) {
		get_peaks_and_troughs(start,c-min);
		start.advance();
	}
	++time;
}

/*
 * CALCAMPSANDPERS
 * calculates amplitudes and periods off of current analysis data
 * arg "c": the cell to analysis amplitudes and periods from
*/
void OscillationAnalysis :: calcAmpsAndPers(int s, int c){
	std::vector<crit_point> crits = peaksAndTroughs[s][c];
    Real peakSum = 0.0, troughSum = 0.0, cycleSum = 0.0;
    int numPeaks = 0, numTroughs = 0, cycles = 0;
	for (std::size_t i = 0; i < crits.size(); ++i){
		if (crits[i].is_peak){
			peakSum+=crits[i].conc;
			++numPeaks;
		}
		else {
			troughSum+=crits[i].conc;
			++numTroughs;
		}
		if (i < 2){
			continue;
		}
		++cycles;
		cycleSum+=(crits[i].time-crits[i-2].time);
	}
	amplitudes[s][c] = ((peakSum/numPeaks)-(troughSum/numTroughs))/2;
	periods[s][c] = cycleSum/cycles;
}
