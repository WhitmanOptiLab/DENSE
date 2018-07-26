/*
 * FINALIZE
 * finds peaks and troughs in final half-window of simulation data
 * precondition: the simulation has finished
*/
template <typename Simulation>
void OscillationAnalysis<Simulation>::finalize() {
    int timeTemp = this->samples;
    for (std::size_t s = 0; s < this->observed_species_.size(); ++s)
    {
        for (Natural c = 0; c < this->max - this->min; ++c){
            this->samples = timeTemp;
            while (windows[s][c].getSize()>=(range_steps/2)&&bst[s][c].size()>0){
                Real removed = windows[s][c].dequeue();
                bst[s][c].erase(bst[s][c].find(removed));
                //std::cout<<"bst size="<<bst[s][c].size()<<'\n';
                checkCritPoint(s, c);
                ++this->samples;
            }
            calcAmpsAndPers(s, c);
        }
    }
}

#include <numeric>

template <typename Simulation>
void OscillationAnalysis<Simulation>::show (csvw * csv_out) {
  Analysis<>::show(csv_out);
  if (csv_out)
  {
      for (Natural c = this->min; c < this->max; ++c) {
        std::vector<Real> avg_peak(this->observed_species_.size());
        for (std::size_t s = 0; s < this->observed_species_.size(); ++s) {
              dense::Natural peak_count = 0;
              auto& x = peaksAndTroughs[s][c];
              avg_peak[s] = std::accumulate(x.begin(), x.end(), 0.0, [&](Real total, crit_point cp) {
                if (cp.is_peak) {
                  ++peak_count;
                  return total + cp.conc;
                }
                return total;
              });
/*
              for (std::size_t pt = 0; pt < peaksAndTroughs[s][c].size(); ++pt)
              {
                  crit_point cp = peaksAndTroughs[s][c][pt];
                  if (cp.is_peak) {
                      avg_peak[s] += cp.conc;
                      ++peak_count;
                  }
              }*/

              if (peak_count != 0) avg_peak[s] /= peak_count;
          }

          *csv_out << "\n# Showing cell " << c << "\nSpecies";
          for (specie_id const& lcfID : this->observed_species_)
              *csv_out << ',' << specie_str[lcfID];

          csv_out->add_div("\navg peak,");
          for (std::size_t s = 0; s < this->observed_species_.size(); ++s)
              csv_out->add_data(avg_peak[s]);

          csv_out->add_div("\navg amp,");
          for (std::size_t s = 0; s < this->observed_species_.size(); ++s)
              csv_out->add_data(amplitudes[s][c]);

          csv_out->add_div("\navg per,");
          for (std::size_t s = 0; s < this->observed_species_.size(); ++s)
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
template <typename Simulation>
void OscillationAnalysis<Simulation>::get_peaks_and_troughs (Simulation const& simulation, int c) {

    for (std::size_t i = 0; i < this->observed_species_.size(); ++i)
    {
        Real added = simulation.get_concentration(c + this->min, this->observed_species_[i]);
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
template <typename Simulation>
void OscillationAnalysis<Simulation>::checkCritPoint (int s, int c) {
	Real mid_conc = windows[s][c].getVal(windows[s][c].getCurrent());
	if (mid_conc == *bst[s][c].rbegin() && mid_conc != *bst[s][c].begin()) {
		addCritPoint(s,c, crit_point{ std::max<Real>(0.0,(this->samples - range_steps/2)*analysis_interval + this->start_time),mid_conc, true });
	}
	else if (mid_conc == *bst[s][c].begin()) {
		addCritPoint(s,c, crit_point{ std::max<Real>(0.0,(this->samples - (range_steps/2))*analysis_interval + this->start_time),mid_conc, false });
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
template <typename Simulation>
void OscillationAnalysis<Simulation>::addCritPoint (int s, int context, crit_point crit) {
	if (peaksAndTroughs[s][context].size() > 0){
		crit_point prev_crit = peaksAndTroughs[s][context].back();
		if (prev_crit.is_peak == crit.is_peak){
			if (crit.is_peak ? (crit.conc >= prev_crit.conc) : (crit.conc <= prev_crit.conc)){
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
template <typename Simulation>
void OscillationAnalysis<Simulation>::update (Simulation& simulation, std::ostream&) {
	for (Natural c = this->min; c < this->max; ++c) {
		get_peaks_and_troughs(simulation, c - this->min);
	}
	++this->samples;
}

/*
 * CALCAMPSANDPERS
 * calculates amplitudes and periods off of current analysis data
 * arg "c": the cell to analysis amplitudes and periods from
*/
template <typename Simulation>
void OscillationAnalysis<Simulation>::calcAmpsAndPers (int s, int c) {
	std::vector<crit_point> crits = peaksAndTroughs[s][c];
    Real peakSum = 0.0, troughSum = 0.0, cycleSum = 0.0;
    int numPeaks = 0, numTroughs = 0, cycles = 0;
	for (std::size_t i = 0; i < crits.size(); ++i) {
    auto& sum = crits[i].is_peak ? peakSum : troughSum;
    auto& count = crits[i].is_peak ? numPeaks : numTroughs;
		sum += crits[i].conc;
		++count;
		if (i < 2){
			continue;
		}
		++cycles;
		cycleSum+=(crits[i].time-crits[i-2].time);
	}
	amplitudes[s][c] = ((peakSum/numPeaks)-(troughSum/numTroughs))/2;
	periods[s][c] = cycleSum/cycles;
}
