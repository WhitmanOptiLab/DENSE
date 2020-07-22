#include <cmath>
#include "num_sim.hpp"
#include "simpson.hpp"
#include "determ.hpp"
#include "trap.hpp"
#include "avg.hpp"
#include "model_impl.hpp"
#include "baby_cl.hpp"
#include <limits>
#include <iostream>
#include <cassert>
/*
dense::Numerical_Integration::Numerical_Integration(int num_delay_rxn, Natural& cell_cnt, Minutes& step_size, Deterministic_Simulation& sim) :
        _intDelays(num_delay_rxn, cell_cnt),
        _step_size{step_size / Minutes{1}}, _j(0), _num_history_steps(4), _baby_cl(sim) {

}

dense::Numerical_Integration::Numerical_Integration(int num_delay_rxn, Natural& cell_cnt, Minutes& step_size, Simpson_Simulation& sim) :
        _intDelays(num_delay_rxn, cell_cnt),
        _step_size{step_size / Minutes{1}}, _j(0), _num_history_steps(4), _baby_cl(sim) {

}

dense::Numerical_Integration::Numerical_Integration(int num_delay_rxn, Natural& cell_cnt, Minutes& step_size, Trapezoid_Simulation& sim) :
        _intDelays(num_delay_rxn, cell_cnt),
        _step_size{step_size / Minutes{1}}, _j(0), _num_history_steps(4), _baby_cl(sim) {

}

dense::Numerical_Integration::Numerical_Integration(int num_delay_rxn, Natural& cell_cnt, Minutes& step_size, Average_Simulation& sim) :
        _intDelays(num_delay_rxn, cell_cnt),
        _step_size{step_size / Minutes{1}}, _j(0), _num_history_steps(4), _baby_cl(sim) {

}
*/
