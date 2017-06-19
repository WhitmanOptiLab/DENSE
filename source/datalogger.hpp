#ifndef DATALOGGER_HPP
#define DATALOGGER_HPP

#include "color.hpp"
#include "param_set.hpp"
#include "model.hpp"
#include "specie.hpp"
#include "cell_param.hpp"
#include "simulation_base.hpp"
#include "simulation_set.hpp"
#include "reaction.hpp"
#include "concentration_level.hpp"
#include "baby_cl.hpp"
#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include "observable.hpp"
#include <iomanip>

using namespace std;

/*
The DataLogger observes Simulation and periodically records the concentrations levels for analysis and record.
*/
class DataLogger : public  Observer, public Observable {	

    const string STR_ALL_SPECIES[NUM_SPECIES] = {
        #define SPECIE(name) #name, 
        #include "specie_list.hpp"
        #undef SPECIE
    };
	
public:
	vector<vector<vector<RATETYPE> > > datalog;
	int last_log_time;

	class DataLog: public ContextBase {
        //FIXME - want to make this private at some point
     	private:
        	int c;
        	DataLogger& logger;

	public:
        	CPUGPU_FUNC
        	DataLog(DataLogger& dl, int context) : logger(dl),c(context){ }
        	CPUGPU_FUNC
        	virtual RATETYPE getCon(specie_id sp) const final {
			return logger.datalog[c][sp][logger.last_log_time];
        	}
        	CPUGPU_FUNC
        	virtual void advance() final { ++c; }
		CPUGPU_FUNC
		virtual bool isValid() const final { return c >= 0 && c < logger.datalog.size(); }
		CPUGPU_FUNC
		virtual void reset() final {c=0;}
	};

	DataLogger();
	/**
	*Constructor for DataLogger
	*sub: subject to observe
	*analysis_gran: interval between concentration level recordings
	*/
	DataLogger(Observable *sub) : Observer(sub) {
		last_log_time = 0;
	}
	
	/**
	*update: overrrides inherited update, called when observable simulation notifies observer
	*/
	virtual void update(ContextBase& start) {
		for (int c = 0; start.isValid(); c++){
			try{
				datalog.at(c);
			}
			catch (const out_of_range& e){
				datalog.emplace_back();
			}
			vector<vector<RATETYPE> >& contextlog = datalog.at(c);
			for (int s = 0; s<NUM_SPECIES; s++){
				specie_id sid = (specie_id) s;
				try{
					contextlog.at(s).push_back(start.getCon(sid));
				}
				catch (const out_of_range& e){
					contextlog.emplace_back();
					contextlog.at(s).push_back(start.getCon(sid));
				}
			}
			start.advance();
		}

		last_log_time++;
		DataLog log(*this,0);
		notify(log);
	}

	virtual void finalize(ContextBase& start){
		DataLog log(*this,0);
		notify(log,true);
	}

/*	
	void reallocateData(int last_relevant_time);
	
	 * Import File To Data -- read data from *.csv file and load it into the Data Logger
	 * ofname: directory/name of file to write to, .csv extension not needed
	
    void importFileToData(const string& pcfFileName)
	{
	    // Context, Specie, Time
	    
	    CSVReader gCSVR(pcfFileName);
	    bool gError = false;
	    
        for (int i = 0; i<species && !gError; i++)
        {
		    for (int j = 0; j<contexts && !gError; j++)
		    {
		        for (int k = 0; k<steps && !gError; k++)
		        {
		            gError = !gCSVR.nextCSVCell(datalog[i][j][k]);
		        }
		    }
	    }
		
		if (gError)
		{
		    cout << color::set(color::RED) << "Failed to import \'" << pcfFileName << "\' to data logger." << color::clear() << endl;
		}
	}*/

/*
    /**	
	 *exportDataToFile: writes logged concentration levels to a file
	 *ofname: directory/name of file to write to, .csv extension not needed
	
    void exportDataToFile(const string& ofname){
	    ofstream outFile;
		outFile.open(ofname);
		for (int c = 0; c < contexts; c++){
			outFile<<"Cell "<<c<<endl<<"Time, ";
			for (int s = 0; s < species; s++){
			    // TODO As of now I have no way of figuring out whether the order in which the specie names are outputted here is the same order in which the data is outputted.
				outFile<<"Specie "<<STR_ALL_SPECIES[s]<<", ";
			}
			outFile<<endl;
			for (int t = 0; t<steps; t++){
				outFile<<t*analysis_interval<<", ";
				for (int s = 0; s<species; s++){
					outFile<<std::setprecision(5)<<datalog[s][c][t]<<", ";
				}
				outFile<<endl;
			}
			outFile<<endl;
		}
		outFile.close();
	}
*/
};

#endif
