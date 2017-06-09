#include "analysis.hpp"

using namespace std;

void BasicAnalysis :: update_averages(){
	
	for (int s=0; s<dl->species; s++){
		for (int c=0; s<dl->contexts; c++){
			for (time; time<(dl->sim->_baby_j[s]-1); time++){
				averages[s] = (averages[s]*time+dl->datalog[s][c][time])/(time+1);
				avgs_by_context[c][s] = (avgs_by_context[c][s]*time+dl->datalog[s][c][time])/(time+1);
			}
		}
	}
}

void BasicAnalysis :: update_minmax(){

	for (int s=0; s<dl->species; s++){
		for (int c=0; s<dl->contexts; c++){
			for (time; time<(dl->sim->_baby_j[s]-1); time++){
				if (dl->datalog[s][c][time] > maxs[s]){
					maxs[s] = dl->datalog[s][c][time];
				}
				if (dl->datalog[s][c][time] < mins[s]){
					mins[s] = dl->datalog[s][c][time];
				}
				if (dl->datalog[s][c][time] > maxs_by_context[c][s]){
					maxs_by_context[c][s] = dl->datalog[s][c][time];
				}
				if (dl->datalog[s][c][time] < mins_by_context[c][s]){
					mins_by_context[c][s] = dl->datalog[s][c][time];
				}
			}
		}
	}
}


void OscillationAnalysis :: get_peaks_and_troughs(){

	int time_temp;

	for (int c=0; c<dl->contexts; c++){

		time_temp = time;
		bool ending = false;
		int loopTo = dl->last_log_time;
		if (dl->last_log_time == dl->steps){
			loopTo+=(range_steps/2);
		}
			
		for (time_temp; time_temp<loopTo; time_temp++){
			
			if (time_temp == dl->last_log_time){
				ending = true;
			}

			if ( windows[c].getSize() == range_steps || ending) {
				RATETYPE removed = windows[c].dequeue();
				bst[c].erase(removed);
                        }
		
			if (!ending){
				RATETYPE added = dl->datalog[s][c][time_temp];	
				windows[c].enqueue(added);
				bst[c].insert(added);
			}

			if ( windows[c].getSize() < range_steps/2 && !ending) {
				continue;
			}

			RATETYPE mid_conc = windows[c].getVal(windows[c].getCurrent()); 

			if (mid_conc==*bst[c].rbegin()){
				addCritPoint(c,true,(time_temp-(range_steps/2))*dl->analysis_interval,mid_conc);
			}
			else if (mid_conc==*bst[c].begin()){
				addCritPoint(c,false,(time_temp-(range_steps/2))*dl->analysis_interval,mid_conc);
			}
		}
	}
	time = time_temp;
}


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


void OscillationAnalysis :: update(){
	get_peaks_and_troughs();
}
