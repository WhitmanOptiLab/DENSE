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

void OscillationAnalysis :: initialize(){
	
	for (int c=0; c<dl->contexts; c++){

		Queue q(range_steps);
		vector<crit_point> v;
		set<RATETYPE> BST;

		windows.push_back(q);
		peaksAndTroughs.push_back(v);
		bst.push_back(BST);

		for (time; time<range_steps/2; time++){
			RATETYPE val = dl->datalog[s][c][time];
			windows[c].enqueue(val);
			bst[c].insert(val);
		}
		for (int t=0; t<range_steps/2; t++){
			RATETYPE check = dl->datalog[s][c][t];
			if (check == *bst[c].end()){
				addCritPoint(c,true,t*dl->analysis_interval,check);
			}
			else if (check == *bst[c].begin()){
				addCritPoint(c,false,t*dl->analysis_interval,check);
			}
			time++;
			RATETYPE next = dl->datalog[s][c][time];
			windows[c].enqueue(next);
			bst[c].insert(next);
		}
	}
}

void OscillationAnalysis :: get_peaks_and_troughs(){

	for (int c=0; c<dl->contexts; c++){
		for (time; time<dl->last_log_time; time++){
			RATETYPE removed = windows[c].dequeue();
			bst[c].erase(removed);
		
			RATETYPE added = dl->datalog[s][c][time];	
			windows[c].enqueue(added);
			bst[c].insert(added);

			RATETYPE mid_conc = windows[c].getVal(windows[c].getMidpoint()); 

			if (mid_conc==*bst[c].end()){
				addCritPoint(c,true,(time-(range_steps/2))*dl->analysis_interval,mid_conc);
			}
			else if (mid_conc==*bst[c].begin()){
				addCritPoint(c,false,(time-(range_steps/2))*dl->analysis_interval,mid_conc);
			}
		}
		if (dl->last_log_time == static_cast<int>(dl->sim->time_total/dl->analysis_interval - 1)){
			int middle = windows[c].getMidpoint();
			for (int l=1; l<range_steps/2; l++){
				RATETYPE removed = windows[c].dequeue();
				bst[c].erase(removed);
				
				RATETYPE check_conc = windows[c].getVal(middle+l);
					
				if (check_conc==*bst[c].end()){
					addCritPoint(c,true,((time-(range_steps/2))+l)*dl->analysis_interval, check_conc);
				}
				else if (check_conc==*bst[c].begin()){
					addCritPoint(c,false,((time-(range_steps/2))+l)*dl->analysis_interval, check_conc);
				}
			}
			cout<<"CRITICAL POINTS GENERATED"<<endl;
		}
	}
}


void OscillationAnalysis :: addCritPoint(int context, bool isPeak, RATETYPE minute, RATETYPE concentration){
	crit_point crit;
	crit.is_peak = isPeak;
	crit.time = minute;
	crit.conc = concentration;
	peaksAndTroughs[context].push_back(crit);
}


void OscillationAnalysis :: update(){
	if (time==0){
		initialize();
	}
	get_peaks_and_troughs();
}
