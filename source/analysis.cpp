#include "analysis.hpp"

using namespace std;

/*
 * UPDATE_AVERAGES
 * arg "start": context iterator to access newest conc level with
 * arg "c": the cell the context inhabits
 * updates average concentration levels across all cells and per cell
*/
void BasicAnalysis :: update_averages(const ContextBase& start, int c){
	
	for (int s=0; s<NUM_SPECIES; s++){
		specie_id sid = static_cast<specie_id>(s);
		averages[s] = (averages[s]*time+start.getCon(sid))/(time+1);
		avgs_by_context[c][s] = (avgs_by_context[c][s]*time+start.getCon(sid))/(time+1);
	}
}

/*
 * UPDATE_MINMAX
 * arg "start": context iterator to access newest conc level with
 * arg "c": the cell the context inhabits
 * updates min and max conc levels across all cells and per cell
*/
void BasicAnalysis :: update_minmax(const ContextBase& start, int c){

	for (int s=0; s<NUM_SPECIES; s++){
		specie_id sid = static_cast<specie_id>(s);
		RATETYPE conc = start.getCon(sid);
		if (conc > maxs[s]){
			maxs[s] = conc;
		}
		if (conc < mins[s]){
			mins[s] = conc;
		}
		if (conc > maxs_by_context[c][s]){
			maxs_by_context[c][s] = conc;
		}
		if (conc < mins_by_context[c][s]){
			mins_by_context[c][s] = conc;
		}
	}
}

/*
 * FINALIZE
 * finds peaks and troughs in final half-window of simulation data
 * arg "start": context iterator to access conc levels with
 * precondition: the simulation has finished
*/
void OscillationAnalysis :: finalize(ContextBase& start){
	
	for (int c=0; start.isValid(); c++){
		for (int t=0; t<range_steps/2; t++){
			RATETYPE removed = windows[c].dequeue();
			bst[c].erase(bst[c].find(removed));
			
			checkCritPoint(c);
		}
		calcAmpsAndPers(c);
		start.advance();
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
		addCritPoint(c,true,(time-range_steps/2)*analysis_interval,mid_conc);
	}
	else if (mid_conc==*bst[c].begin()){
		addCritPoint(c,false,(time-(range_steps/2))*analysis_interval,mid_conc);
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

	for (int c=0; start.isValid(); c++){		
		if (time==0){
			Queue q(range_steps);
			vector<crit_point> v;
			multiset<RATETYPE> BST;

			windows.push_back(q);
			peaksAndTroughs.push_back(v);
			bst.push_back(BST);
				
			amplitudes.push_back(0);
			periods.push_back(0);
			numPeaks.push_back(0);
			numTroughs.push_back(0);
			cycles.push_back(0);
			peakSum.push_back(0);
			troughSum.push_back(0);
			cycleSum.push_back(0);
			crit_tracker.push_back(0);
		}
		
		get_peaks_and_troughs(start,c);
		calcAmpsAndPers(c);
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
	if (crit_tracker[c] < crits.size()){
		for (crit_tracker[c];crit_tracker[c]<crits.size();crit_tracker[c]++){
			if (crits[crit_tracker[c]].is_peak){
				peakSum[c]+=crits[crit_tracker[c]].conc;
				numPeaks[c]++;
			}
			else{
				troughSum[c]+=crits[crit_tracker[c]].conc;
				numTroughs[c]++;
			}
			if (crit_tracker[c]<2){
				continue;
			}
			cycles[c]++;
			cycleSum[c]+=(crits[crit_tracker[c]].time-crits[crit_tracker[c]-2].time);
		}
		amplitudes[c] = ((peakSum[c]/numPeaks[c])-(troughSum[c]/numTroughs[c]))/2;
		periods[c] = cycleSum[c]/cycles[c];
	}
}

