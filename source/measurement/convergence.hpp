#ifndef ANLYS_BASIC_HPP
#define ANLYS_BASIC_HPP

#include "base.hpp"

#include <vector>
#include <stdexcept>
#include <cmath>

/*
* Subclass of Analysis superclass
* - records a predicted convergence vlaue
* - records if the data points converge or not
*/

template <typename simulation>
class ConvergenceAnalysis : public Analysis<Simulation>{
  public:
    //structure for converges, convergence_values, convergence_start
    struct Convergence {
        Real value;
        Real start;
        void merge_results(Real newValue, Real newStart) {
            if (isnan(value)) {
                start = newStart;
            }
          value = newValue;
        }
    };

    ConvergenceAnalysis(Real interval, Real windowSize, Real threshold, std::vector<Species> const& pcfSpecieOption, 
                        std::pair<dense::Natural, dense::Natural> cell_range, 
                        std::pair<Real, Real> time_range = { 0, std::numeric_limits<Real>::infinity() }) :
        Analysis<Simulation>(pcfSpecieOption, cell_range, time_range), 
        window_size(windowSize), threshold(threshold), finalize(false), 
        converges(Analysis<>::observed_species_.size(), false),
        window_steps(windowSize/interval), analysis_interval(interval)
    {
        auto cell_count = Analysis<>::max - Analysis<>::min;
        for (std::size_t i = 0; i < Analysis<>::observed_species_.size(); ++i) {
            windows.emplace_back(Analysis<>::observed_species_.size(), Queue<Real>(window_steps));
            convergences.emplace_back(Analysis<>::observed_species_.size(), Convergence(NAN, 0));
        }
    }

    //whenever the object has new data to be read
    //add concentration to queue
    //check for
    void update(Simulation& simulation, std::ostream& log) override {
        
        //not sure yet how this type of update would differ from the other analysis types
        for (Natural c = this->min; c < this->max; ++c){
            for (std::size_t i =0; i < this-> pcfSpecieOption.size(); ++i){
                Real concentration = simulation.get_concentration(c, this->pcfSpecieOption[i]);
                //FIX
                windows[c][i].dequeue();
                windows[c][i].enqueue(concentration);
                convergences[c][i].merge_results(check_convergence(windows[c][i], concentration),
                                                 simulation.age());
            }
        }
    }

    Real check_convergence(const Queue<Real>& window, Real asymptote) {
        if ( window.getSize() != window_size ) {
            return NAN;
        }

        for (std::size_t i = 0; i < window.getSize(); i++) {
            if (abs( (window.getVal(i)/asymptote) - 1.0 ) > threshold) {
                return NAN;
            }
        }

        return asymptote;
    }

    
    bool windowCheck(Real val){
        if (val >= convergence_value){
            convergence = true;
            convergence_bools.push_back(convergence);
        } else {
            
        }
    }

    //copy constructor
    ConvergenceAnalysis* clone() const override{
        return new auto(*this);
    }

    //call once your done collecting data
    void finalize() override {
        if(!finalize){
            finalize = true;
        }
        for (std::size_t i =0; i < this-> pcfSpecieOption.size(); ++i){
            converges[i] = true;
            for (Natural c = this->min; c < this->max; ++c){
                if (isnan(convergences[c][i].value)) {
                    converges[i] = false;
                    break;
                }
            }
        }
    }

    //analysis overview
    //what we give back to the user
    //must provide where convergence happens and when
    Details get_details() override {
        Details detail;
        for (std::size_t i =0; i < this-> pcfSpecieOption.size(); ++i){
            detail.means.push_back( converges[i] ? 1.0 : 0.0 );
        }
        for (Natural c = this->min; c < this->max; ++c){
            std::vector<Real> values;
            std::vector<Real> starts;
            for (std::size_t i =0; i < this-> pcfSpecieOption.size(); ++i){
              value.push_back(convergences[c][i].value);
              starts.push_back(convergences[c][i].start);
            }
            detail.other_details.push_back(values);
            detail.other_details.push_back(starts);
        }
    }

  private:
    std::vector<std::vector<Queue<Real>>> windows;
    std::vector<std::vector<Convergence>> convergences;
    Real window_size;
    DENSE::Natural window_steps;
    Real threshold;
    bool finalize;
    std::vector<bool> converges;
}

#endif
