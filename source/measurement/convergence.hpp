#ifndef ANLYS_BASIC_HPP
#define ANLYS_BASIC_HPP

#include "base.hpp"

#include <vector>
#include <stdexcept>

/*
* Subclass of Analysis superclass
* - records a predicted convergence vlaue
* - records if the data points converge or not
*/

template <typename simulation>
class ConvergenceAnalysis : public Analysis<Simulation>{
    public:
    //structure for converges, convergence_values, convergence_start
    struct{
        Real convergence_value;
        Real convergence_start;
        bool converges;
    }
    ConvergenceAnalysis(Real windowSize, Real thresHold, std::vector<Species> const& pcfSpecieOption, std::pair<dense::Natural, dense::Natural> cell_range, std::pair<Real, Real> time_range = { 0, std::numeric_limits<Real>::infinity() }) :
    Analysis<Simulation>(pcfSpecieOption, cell_range, time_range), range_steps(range/interval), analysis_interval(interval){
            auto cell_count = Analysis<>::max - Analysis<>::min;
            for (std::size_t i = 0; i < Analysis<>::observed_species_.size(); ++i){
                
                
            }
        finalize = false;
        converges = false;
        }

        //whenever the object has new data to be read
        //add concentration to queue
        //check for
        void update(Simulation& simulation, std::ostream& log) override {
            
            //not sure yet how this type of update would differ from the other analysis types
            for (Natural c = this->min; c < this->max; ++c){
                for (std::size_t i =0; i < this-> pcfSpecieOption.size(); ++i){
                    Real concentration = simulation.get_concentration(c, this->pcfSpecieOption[i]);
                    windows[i].enqueue(concentration);
                    converges = windowCheck(concentration);
                }
                //not sure what to pass this into
            }
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
        }
    
        //analysis overview
        //what we give back to the user
        //must provide where convergence happens and when
        Details get_details() override {
            std::vector<Real> times;
            
        }
    private:
        std::vector<std::vector<Queue<Real>>> windows;
        //another 2d vector for every species (convergence state)
        std::vector<std::vector<bool>> convergence_bools;
        Real window_size;
        Real threshold;
    
        bool finalize;
    
}

#include "basic.ipp"

#endif
