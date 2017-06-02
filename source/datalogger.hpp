#ifndef DATALOGGER_HPP
#define DATALOGGER_HPP

#include "param_set.hpp"
#include "model.hpp"
#include "specie.hpp"
#include "cell_param.hpp"
#include "simulation.hpp"
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
class DataLogger : public  Observer {
	
	int analysis_interval;
	
	RATETYPE*** datalog;

	int last_log_time;
	int species;
	int contexts;
	int steps;
	
	simulation* sim;	

	public:

	DataLogger();
	DataLogger(int = 1);

	/**
	*Constructor for DataLogger
	*sub: simulation to observe
	*analysis_gran: interval between concentration level recordings
	*/
	DataLogger(simulation *sub, int analysis_gran) : Observer(sub) {
		
		sim = sub;
		analysis_interval = analysis_gran;

		species = NUM_SPECIES;
		contexts = sim->_cells_total;
		steps = sim->time_total/analysis_interval;

		datalog = new RATETYPE**[species];
		for (int i = 0; i < species; i++){
			datalog[i] = new RATETYPE*[contexts];
			for (int j=0; j<contexts; j++){
				datalog[i][j] = new RATETYPE[steps];
			}
		}
		
		last_log_time = 0;
	}
	
	/**
	*update: overrrides inherited update, called when observable simulation notifies observer
	*/
	void update() {
		for (int s = 0; s<species; s++){
			for (int c = 0; c<contexts; c++){
				datalog[s][c][last_log_time] = sim->_baby_cl[s][sim->_baby_j[s]-1][c];
			}
		}
		last_log_time++;
	}
	
	void testDataLogger();
	
	void reallocateData(int last_relevant_time);

	/**
	*exportDataToFile: writes logged concentration levels to a file
	*outFile: given file object to write to, named "dataOut"
	*/
	void exportDataToFile(ofstream& outFile){
		outFile.open("dataOut.txt");
		for (int c = 0; c < contexts; c++){
			outFile<<"    Cell "<<c<<endl;
			for (int s = 0; s < species; s++){
				outFile<<"    Specie "<<s;
			}
			outFile<<endl;
			for (int t = 0; t<steps; t++){
				outFile<<"Time "<<t*analysis_interval;
				for (int s = 0; s<species; s++){
					outFile<<std::setprecision(5)<<datalog[s][c][t]<<"    ";
				}
				outFile<<endl;
			}
		}
	}
	
	void importFileToData(ifstream inFile);
};

#endif
