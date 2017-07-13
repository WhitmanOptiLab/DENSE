#include "oscillation.hpp"

using namespace std;


/*
 * FINALIZE
 * finds peaks and troughs in final half-window of simulation data
 * precondition: the simulation has finished
*/
void OscillationAnalysis :: finalize(){
	
	for (int c=0; c<max-min; c++){
        cout<<"c="<<c<<endl;
		while (windows[c].getSize()>1&&bst[c].size()>0){
            RATETYPE removed = windows[c].dequeue();
            bst[c].erase(bst[c].find(removed));
            //cout<<"bst size="<<bst[c].size()<<endl;
			checkCritPoint(c);
		}
		calcAmpsAndPers(c);
	}
}

/*
 * GET_PEAKS_AND_TROUGHS
 * advances the local range window and identifies critical points
 * arg "start": context iterator to access conc levels with
 * arg "c": the cell the context inhabits
*/ 
void OscillationAnalysis :: get_peaks_and_troughs(const ContextBase& start, int c){

	RATETYPE added = start.getCon(s);
	windows[c].enqueue(added);
	bst[c].insert(added);
	if ( windows[c].getSize() == range_steps + 1) {
		RATETYPE removed = windows[c].dequeue();
		bst[c].erase(bst[c].find(removed));
 	}
	if ( windows[c].getSize() < range_steps/2) {
		return;
	}
	checkCritPoint(c);
}

/*
 * CHECKCRITPOINT
 * determines if a particular specie conc level in a particular cell is a peak, trough, or neither
 * arg "c": the cell this concentration level is found in
*/
void OscillationAnalysis :: checkCritPoint(int c){
	RATETYPE mid_conc = windows[c].getVal(windows[c].getCurrent());
	if (mid_conc==*bst[c].rbegin()){
		addCritPoint(c,true,std::max(0.0,(time-range_steps/2)*analysis_interval+start_time),mid_conc);
	}
	else if (mid_conc==*bst[c].begin()){
		addCritPoint(c,false,std::max(0.0,(time-(range_steps/2))*analysis_interval+start_time),mid_conc);
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
void OscillationAnalysis :: addCritPoint(int context, bool isPeak, RATETYPE minute, RATETYPE concentration){
	crit_point crit;
	crit.is_peak = isPeak;
	crit.time = minute;
	crit.conc = concentration;
	
	if (peaksAndTroughs[context].size() > 0){
		crit_point prev_crit = peaksAndTroughs[context].back();
		if (prev_crit.is_peak == crit.is_peak){
			if ((crit.is_peak && crit.conc>=prev_crit.conc)||(!crit.is_peak&&crit.conc<=prev_crit.conc)){
				peaksAndTroughs[context].back() = crit;
			}
		}
		else{
			peaksAndTroughs[context].push_back(crit);
		}
	}else{
		peaksAndTroughs[context].push_back(crit);
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
void OscillationAnalysis :: update(ContextBase& start){
	for (int c=min; c<max && start.isValid(); c++){
		get_peaks_and_troughs(start,c-min);
		start.advance();
	}
	time++;
}

/*
 * CALCAMPSANDPERS
 * calculates amplitudes and periods off of current analysis data
 * arg "c": the cell to analysis amplitudes and periods from
*/
void OscillationAnalysis :: calcAmpsAndPers(int c){
	vector<crit_point> crits = peaksAndTroughs[c];
    RATETYPE peakSum = 0.0, troughSum = 0.0, cycleSum = 0.0;
    int numPeaks = 0, numTroughs = 0, cycles = 0;
	for (int i=0; i<crits.size(); i++){
		if (crits[i].is_peak){
			peakSum+=crits[i].conc;
			numPeaks++;
		}
		else{
			troughSum+=crits[i].conc;
			numTroughs++;
		}
		if (i<2){
			continue;
		}
		cycles++;
		cycleSum+=(crits[i].time-crits[i-2].time);
	}
	amplitudes[c] = ((peakSum/numPeaks)-(troughSum/numTroughs))/2;
	periods[c] = cycleSum/cycles;
}

