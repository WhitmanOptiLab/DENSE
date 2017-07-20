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
	BasicAnalysis(Observable *dLog, const specie_vec& pcfSpecieOption, csvw* pnFileOut,
            int mn, int mx, RATETYPE startT, RATETYPE endT) 
        : Analysis(dLog,pcfSpecieOption,pnFileOut,mn,mx,startT,endT)
    {
        for (int c=min; c<max; c++){
            avgs_by_context.emplace_back();
		    mins_by_context.emplace_back();
		    maxs_by_context.emplace_back();
		    for (const specie_id& lcfID : ucSpecieOption){
			    mins_by_context[c].push_back(9999);
		    	maxs_by_context[c].push_back(0);
			    avgs_by_context[c].push_back(0);
			    if (c==min){
					mins.push_back(9999);
					maxs.push_back(0);
				  	averages.push_back(0);
				}
            }
        }
    }

	/*
	* Update: repeatedly called by observable to notify that there is more data
	* - arg ContextBase& start: reference to iterator over concentrations
	* - precondition: start.isValid() is true.
	* - postcondition: start.isValid() is false.
	* - update is overloaded virtual function of Observer
	*/
	void update(ContextBase& start){
		for (int c = min; c<max; c++){
			update_averages(start,c);
			update_minmax(start,c);
			start.advance();
		}
		time++;
	}

    /*
	// Test: prints output.
	void test(){
		for (int s=0; s<averages.size(); s++){
			cout<<"Specie "<<s<<endl<<"average="<<averages[s]<<endl;
			cout<<"minimum="<<mins[s]<<endl;
			cout<<"maximum="<<maxs[s]<<endl<<endl;
		}
	}
    */

	/* Finalize: overloaded virtual function of observer
	   - must be called to produce correct average values
	*/	
	void finalize();
	
	virtual ~BasicAnalysis() {}
};


#endif
