#ifndef IN_MEM_LOG_HPP
#define IN_MEM_LOG_HPP

#include "base.hpp"

class in_memory_log{
  public:

    in_memory_log(std::vector<Species> const& pcfSpecieOption, 
                        std::pair<dense::Natural, dense::Natural> cell_range, 
                        std::pair<Real, Real> time_range = { 0, std::numeric_limits<Real>::infinity() }) :
        Analysis<Simulation>(pcfSpecieOption, cell_range, time_range),  
        finalized(false), 
    {
        //do we need a true false bool for whether it is analysis or sim?
    }

    //copy constructor
    in_memory_log* clone() const override{
        return new auto(*this);
    }

    //fix
    //call once your done collecting data
    void finalize() override {
        if(!finalized){
            finalized = true;
        }
        for (std::size_t i =0; i < Analysis<>::observed_species_.size(); ++i){
            converges[i] = true;
            for (Natural c = this->min; c < this->max; ++c){
                if (isnan(convergences[c][i].value)) {
                    converges[i] = false;
                    break;
                }
            }
        }
    }

    //fix
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

    //fix
    Details get_details() override {
        Details detail;
        for (std::size_t i =0; i < Analysis<>::observed_species_.size(); ++i){
            detail.concs.push_back( converges[i] ? 1.0 : 0.0 );
        }
        for (Natural c = this->min; c < this->max; ++c){
            std::vector<Real> values;
            std::vector<Real> starts;
            for (std::size_t i =0; Analysis<>::observed_species_.size(); ++i){
              values.push_back(convergences[c][i].value);
              starts.push_back(convergences[c][i].start);
            }
        }
        return detail;
    }

    //get concentration of each species at a time t
    void update(Simulation& simulation, std::ostream&) override {
        for (Natural cell_no = this->min; cell_no < this->max; ++cell_no) {
            for (std::size_t i = 0; i < this->observed_species_.size(); ++i) {
  		        Real concentration = simulation.get_concentration(cell_no, this->observed_species_[i]);
                concentrations[cell_no][i] = concentration;
  	        }
        }
    }
     
    void age_by (Minutes duration) {

    }

  private:
    std::vector<std::vector<Real>> concentrations; 
    bool finalized;


};
