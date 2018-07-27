/*
Stochastically ranked evolutionary strategy sampler for zebrafish segmentation
Copyright (C) 2013 Ahmet Ay, Jack Holland, Adriana Sperlea, Sebastian Sangervasi

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
sres.hpp contains function declarations for sres.cpp.
*/

#ifndef SRES_HPP
#define SRES_HPP

#include "core/parameter_set.hpp"

#if defined(MPI)
	#include "libsres-mpi/ESES.hpp"
  #include "libsres-mpi/sharefunc.hpp"
#else
	#include "libsres/ESES.hpp"
  #include "libsres/sharefunc.hpp"
#endif

#include <cassert>
#include <vector>

class SRES {
 public:
  typedef std::vector<double> (*SRES_Scorer) (const std::vector<Parameter_Set>&);

 private:
  //SRES interface members
	ESParameter* param;
	ESPopulation* population;
	ESStatistics* stats;
	double pf;
	const Parameter_Set& lowerBounds;
  const Parameter_Set& upperBounds;
  int popsize, parentsize, ngenerations;
  double lb[NUM_PARAMS];
  double ub[NUM_PARAMS];

  //DDESim interface members
  SRES_Scorer score_fcn;
 public:
	SRES (int population_size, int num_parents, int num_generations,
	      const Parameter_Set& lBounds, const Parameter_Set& uBounds, SRES_Scorer scorer,
	      int seed = 0) :
	    param(nullptr), population(nullptr), stats(nullptr), pf(0),
	    lowerBounds(lBounds), upperBounds(uBounds), popsize(population_size),
	    parentsize(num_parents), ngenerations(num_generations), score_fcn(scorer)
	{
    assert(popsize > parentsize && "ERROR: sres parent size must be smaller than the population size.");
	  //Create array of double-precision parameters from the Real parameter sets
    for (int i = 0; i < NUM_PARAMS; ++i) {
      lb[i] = lowerBounds.data()[i];
      ub[i] = upperBounds.data()[i];
    }

    ESInitial(seed, //random seed
        &param, //Parameter set to be filled
        nullptr, //Optional transform function
        [this](double** pop, double* scores, double** constr, int popc) {
          simulate_population(popc, pop, scores);
        }, //Fitness evaluation callback
        esDefESSlash, //ES process, esDefESPlus/esDefESSlash
        0, //constraint
        NUM_PARAMS, //dim : number of parameters
        ub, //upper search bounds
        lb, //lower search bounds
        parentsize, //Number of parents for next generation
        popsize, //Population size
        ngenerations, //Number of generations
        esDefGamma, //Default gamma value
        esDefAlpha, //Default alpha value
        esDefVarphi, //Default expected rate of convergence
        0, //retry
        &population, //Population to fill
        &stats); //Statistics
	}

	~SRES() {
    ESDeInitial(param, population, stats);
  }

  //Retrieve the parameter sets of the current population
  void nextGeneration();
  void simulate_population(int popc, double** population, double* scores);
};

#endif
