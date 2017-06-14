#ifndef DATALOGGER_HPP
#define DATALOGGER_HPP

#include "color.hpp"
#include "param_set.hpp"
#include "model.hpp"
#include "specie.hpp"
#include "cell_param.hpp"
#include "simulation.hpp"
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
	int last_log_time;
	int species;
	int contexts;
	int steps;
	
	RATETYPE analysis_interval;
	simulation* sim;	
	RATETYPE*** datalog;
	
	DataLogger();
	DataLogger(int = 1);

	/**
	*Constructor for DataLogger
	*sub: simulation to observe
	*analysis_gran: interval between concentration level recordings
	*/
	DataLogger(simulation *sub, RATETYPE analysis_gran) : Observer(sub) {
		sim = sub;
		analysis_interval = analysis_gran;

		species = NUM_SPECIES;
		contexts = sim->_cells_total;
		steps = sim->time_total/analysis_interval;
		cout<<"steps = "<<steps<<"  time_total = "<<sim->time_total<<"  analysis_interval = "<<analysis_interval<<endl;

		datalog = new RATETYPE**[species];
		for (int i = 0; i < species; i++){
			datalog[i] = new RATETYPE*[contexts];
			for (int j=0; j<contexts; j++){
				datalog[i][j] = new RATETYPE[steps];
			}
		}
		
		last_log_time = 0;
	}

	~DataLogger() {
		for (int i = 0; i < species; i++) {
			for (int j = 0; j < contexts; j++){
				delete[] datalog[i][j]; 
			}
			delete[] datalog[i];
		}
		delete[] datalog;
	}
	
	
	/**
	*update: overrrides inherited update, called when observable simulation notifies observer
	*/
	virtual void update(ContextBase& start) {
	  for (int c = 0; c<contexts; c++){
	  	for (int s = 0; s<species; s++){
        datalog[s][c][last_log_time] = start.getCon(s);
			}
      start.advance();
		}
		last_log_time++;
		if (last_log_time%100==0){
			notify();
		}
	}
	
	void testDataLogger();
	
	void reallocateData(int last_relevant_time);
	
	/**
	 * Import File To Data -- read data from *.csv file and load it into the Data Logger
	 * ofname: directory/name of file to write to, .csv extension not needed
	*/
	/*void importFileToData(const string& pcfFileName)
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

	/**
	*exportDataToFile: writes logged concentration levels to a file
	*ofname: directory/name of file to write to, .csv extension not needed
	*/
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
};

#endif
