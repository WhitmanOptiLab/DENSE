#ifndef ANLYS_BASIC_HPP
#define ANLYS_BASIC_HPP

#include "base.hpp"
#include <vector>

/*
* Subclass of Analysis superclass
* - records overall mins and maxs for each specie
* - records mins and maxs for each specie per cell
* - records overall specie averages
* - records specie averages per cell
*/
class BasicAnalysis : public Analysis {

private:
	vector<RATETYPE> averages;
	vector<RATETYPE> mins;
	vector<RATETYPE> maxs;
	vector<vector<RATETYPE> > avgs_by_context;
	vector<vector<RATETYPE> > mins_by_context;
	vector<vector<RATETYPE> > maxs_by_context;

	void update_averages(const ContextBase& start,int c);
	void update_minmax(const ContextBase& start,int c);

public:
	BasicAnalysis(Observable *dLog) : Analysis(dLog) {
	}

	/*
	* Update: repeatedly called by observable to notify that there is more data
	* - arg ContextBase& start: reference to iterator over concentrations
	* - precondition: start.isValid() is true.
	* - postcondition: start.isValid() is false.
	* - update is overloaded virtual function of Observer
	*/
	void update(ContextBase& start){
		for (int c = 0; start.isValid(); c++){
			if (time==0){
				avgs_by_context.emplace_back();
				mins_by_context.emplace_back();
				maxs_by_context.emplace_back();
				for (int s = 0; s<NUM_SPECIES; s++){
					mins_by_context[c].push_back(9999);
					maxs_by_context[c].push_back(0);
					avgs_by_context[c].push_back(0);
					if (c==0){
						mins.push_back(9999);
						maxs.push_back(0);
						averages.push_back(0);
					}
				}
			}
			update_averages(start,c);
			update_minmax(start,c);
			start.advance();
		}
		time++;
	}

	// Test: prints output.
	void test(){
		for (int s=0; s<averages.size(); s++){
			cout<<"Specie "<<s<<endl<<"average="<<averages[s]<<endl;
			cout<<"minimum="<<mins[s]<<endl;
			cout<<"maximum="<<maxs[s]<<endl<<endl;
		}
	}

	/* Finalize: overloaded virtual function of observer
	   - does nothing.
	*/	
	void finalize(ContextBase& start){
	}
	
};


#endif
