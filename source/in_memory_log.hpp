#ifndef IN_MEM_LOG_HPP
#define IN_MEM_LOG_HPP

#include "base.hpp"

template <typename Sim>
class in_memory_log : public Analysis<Sim>, public Simulation{
  public:

    in_memory_log(std::vector<Species> const& pcfSpecieOption, 
                        std::pair<dense::Natural, dense::Natural> cell_range, 
                        std::pair<Real, Real> time_range = { 0, std::numeric_limits<Real>::infinity() }),
                        std::chrono::duration<Real, std::chrono::minutes::period> analysis_interval :
        Analysis<Sim>(pcfSpecieOption, cell_range, time_range),  
        finalized(false),
        iSpecieVec(pcfSpecieOption),
        analysis_interval(analysis_interval)
    {
        time = 0;
    }

    //copy constructor
    ConvergenceAnalysis* clone() const override{
        return new auto(*this);
    }

    //call once your done collecting data
    void finalize() override {
        if(!finalized){
            finalized = true;
        }
        time = 0;
    }

    void show (csvw *csv_out = nullptr) override{
        Analysis<>::show(csv_out);
        if(csv_out){
            for (Natural c = this->min; c < this->max; ++c) {

                *csv_out << "\n# Showing cell " << c << "\nSpecies";
                 for (specie_id const& lcfID : this->observed_species_)
                     *csv_out << ',' << specie_str[lcfID];
                
                csv_out->add_div("\nconvergence value,");
                for(std::size_t i =0; i < Analysis<>::observed_species_.size(); ++i)
                    csv_out->add_data(convergences[c][i].value);
                

            }
        }
    }

    /*
    Details get_details() override {
        Details detail;
        for (Natural c = this->min; c < this->max; ++c){
            std::vector<Real> values;
            for (std::size_t i =0; Analysis<>::observed_species_.size(); ++i){
              values.push_back(concentrations[c][i].value);
            }
            detail.other_details.push_back(values);
        }
        return detail;
    }
    */

    //get concentration of each species at a time t
    void update(Sim& simulation, std::ostream&) override {
        for (Natural cell_no = this->min; cell_no < this->max; ++cell_no) {
            for (std::size_t i = 0; i < this->observed_species_.size(); ++i) {
  		        Real concentration = simulation.get_concentration(cell_no, this->observed_species_[i]);
                concentrations[cell_no][i][time] = concentration;
  	        }
        }
        time++;
    }

    Real get_concentration(dense::Natural cell, specie_id species, dense::Natural) const {
      return concentrations[cell][species][time];
    }

    int getCellStart()
    {
        return iCellStart;
    }

    int getCellEnd()
    {
        return iCellEnd;
    }

    Minutes to_minutes(dense:Natural time){
        return (Minutes)(time * analysis_interval)
    }

    dense::Natural to_natural(Minutes time){
        return (dense::Natural)(time / analysis_interval)
    }
     
    void age_by (Minutes duration) {
        Minutes total_time = to_minutes(time) + duration;
        time = to_natural(total_time);
        /*
        Minutes stopping_time = age() + duration;
        while (age() < stopping_time) {
            iRate.resize(cell_count());
            for (dense::Natural cell = iCellStart; cell < iCellEnd; ++cell) {
                age_by((iTimeCol ? Minutes{ csvr::next<Real>() } : stopping_time) - age());
                for (auto & species : iSpecieVec) {
                    iRate[cell][species] = csvr::next<Real>();
                }
            }
            iRate.clear();
        }
        return age();
        */
    }

  private:
    std::vector<std::vector<std::vector<Real>>> concentrations;
    dense::Natural time;
    bool finalized;
    std::chrono::duration<Real, std::chrono::minutes::period> analysis_interval;

    std::vector<Species> iSpecieVec;
    Natural iCellStart, iCellEnd;
};
