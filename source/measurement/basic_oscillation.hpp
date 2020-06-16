//
//
//  
//
//  Created by Myan Sudharsanan on 6/14/20.
//

#ifndef basic_oscillation_h
#define basic_oscillation_h

#include "base.hpp"
//#include "oscillation.hpp"

#include <vector>
#include <set>
#include <string>
#include "core/queue.hpp"
#include <numeric>

template <typename Simulation>
class BasicOscillationAnalysis : public Analysis<Simulation> {
    
private:
    struct crit_point {
        Real time;
        Real conc;
        bool is_peak;
    };
    
    bool vectors_assigned;
    bool finalized;
    Details detail;
    
    // Outer-most vector is "for each specie in observed_species_"
    
    std::vector<std::vector<Queue<Real>>> windows;
    
    std::vector<std::vector<std::vector<crit_point>>> peaksAndTroughs;
    
    const int range_steps = 5;
    Real analysis_interval;
    
    //std::vector<std::vector<Real>> data;
    
    std::vector<std::vector<Real>> amplitudes;
    std::vector<std::vector<Real>> periods;
    
    // s: std::vector<Species> index
    //template <typename Simulation>
    void addCritPoint (int s, int context, crit_point crit) {
        if (peaksAndTroughs[s][context].size() > 0){
            crit_point prev_crit = peaksAndTroughs[s][context].back();
            if (prev_crit.is_peak == crit.is_peak){
                if (crit.is_peak ? (crit.conc >= prev_crit.conc) : (crit.conc <= prev_crit.conc)){
                    peaksAndTroughs[s][context].back() = crit;
                }
            }
            else {
                peaksAndTroughs[s][context].push_back(crit);
            }
        } else {
            peaksAndTroughs[s][context].push_back(crit);
        }
    }
    
    //template <typename Simulation>
    void calcAmpsAndPers (int s, int c) {
        std::vector<crit_point> crits = peaksAndTroughs[s][c];
        Real peakSum = 0.0, troughSum = 0.0, cycleSum = 0.0;
        int numPeaks = 0, numTroughs = 0, cycles = 0;
        for (std::size_t i = 0; i < crits.size(); ++i) {
            auto& sum = crits[i].is_peak ? peakSum : troughSum;
            auto& count = crits[i].is_peak ? numPeaks : numTroughs;
            sum += crits[i].conc;
            ++count;
            if (i < 2){
                continue;
            }
            ++cycles;
            cycleSum+=(crits[i].time-crits[i-2].time);
        }
        amplitudes[s][c] = ((peakSum/numPeaks)-(troughSum/numTroughs))/2;
        periods[s][c] = cycleSum/cycles;
    }
    
    //template <typename Simulation>
    void get_peaks_and_troughs (Simulation const& simulation, int c) {
        
        for (std::size_t i = 0; i < this->observed_species_.size(); ++i)
        {
            Real added = simulation.get_concentration(c + this->min, this->observed_species_[i]);
            windows[i][c].enqueue(added);
            //bst[i][c].insert(added);
            if ( windows[i][c].getSize() == range_steps + 1) {
                windows[i][c].dequeue();
                //bst[i][c].erase(bst[i][c].find(removed));
            }
            if ( windows[i][c].getSize() < range_steps/2) {
                return;
            }
            checkCritPoint(i, c);
        }
    }
    
    //template <typename Simulation>
    void checkCritPoint (int s, int c) {
        Real mid_conc = windows[s][c].getVal(windows[s][c].getCurrent());
        if (mid_conc == getMaximum(s, c)){
            addCritPoint(s,c, crit_point{ std::max<Real>(0.0,(this->samples - range_steps/2)*analysis_interval + this->start_time),mid_conc, true });
        } else if (mid_conc == getMinimum(s, c)) {
            addCritPoint(s,c, crit_point{ std::max<Real>(0.0,(this->samples - (range_steps/2))*analysis_interval + this->start_time),mid_conc, false });
        }
        /*if (mid_conc == *bst[s][c].rbegin() && mid_conc != *bst[s][c].begin()) {
         addCritPoint(s,c, crit_point{ std::max<Real>(0.0,(this->samples - range_steps/2)*analysis_interval + this->start_time),mid_conc, true });
         }
         else if (mid_conc == *bst[s][c].begin()) {
         addCritPoint(s,c, crit_point{ std::max<Real>(0.0,(this->samples - (range_steps/2))*analysis_interval + this->start_time),mid_conc, false });
         }*/
    }
    
    //template <typename Simulation>
    Real getMinimum(int i, int n){
        Real min = windows[i][n].getVal(windows[i][n].getCurrent());;
        for(Natural c = 0; c < range_steps; ++c){
            Real comp = windows[i][n].getVal(c);
            if(comp <= min){
                min = comp;
            }
        }
        return min;
    }
    
    //template <typename Simulation>
    Real getMaximum(int i, int n){
        Real max = windows[i][n].getVal(windows[i][n].getCurrent());;
        for(Natural c = 0; c < range_steps; ++c){
            Real comp = windows[i][n].getVal(c);
            if(comp >= max){
                max = comp;
            }
        }
        return max;
    }
    
public:
    BasicOscillationAnalysis(Real interval,
                        std::vector<Species> const& pcfSpecieOption,
                        std::pair<dense::Natural, dense::Natural> cell_range,
                        std::pair<Real, Real> time_range = { 0, std::numeric_limits<Real>::infinity() }) :
    Analysis<Simulation>(pcfSpecieOption, cell_range, time_range), analysis_interval(interval)
    {
        auto cell_count = Analysis<>::max - Analysis<>::min;
        for (std::size_t i = 0; i < Analysis<>::observed_species_.size(); ++i)
        {
            windows.emplace_back(cell_count, Queue<Real>(range_steps));
            peaksAndTroughs.emplace_back(cell_count);
            //data.emplace_back(cell_count);
            amplitudes.emplace_back(cell_count, 0.0);
            periods.emplace_back(cell_count, 0.0);
        }
        finalized = false;
    }

    BasicOscillationAnalysis* clone() const override {
        return new auto(*this);
    }
    
    
    //template <typename Simulation>
    void show(csvw * csv_out) override{
        Analysis<>::show(csv_out);
        if (csv_out)
        {
            for (Natural c = this->min; c < this->max; ++c) {
                std::vector<Real> avg_peak(this->observed_species_.size());
                for (std::size_t s = 0; s < this->observed_species_.size(); ++s) {
                    dense::Natural peak_count = 0;
                    auto& x = peaksAndTroughs[s][c];
                    avg_peak[s] = std::accumulate(x.begin(), x.end(), 0.0, [&](Real total, crit_point cp) {
                        if (cp.is_peak) {
                            peak_count = peak_count + 1;
                            return total + cp.conc;
                        }
                        return total;
                    });
                    /*
                     for (std::size_t pt = 0; pt < peaksAndTroughs[s][c].size(); ++pt)
                     {
                     crit_point cp = peaksAndTroughs[s][c][pt];
                     if (cp.is_peak) {
                     avg_peak[s] += cp.conc;
                     ++peak_count;
                     }
                     }*/
                    
                    if (peak_count != 0) avg_peak[s] /= Real(peak_count);
                }
                
                *csv_out << "\n# Showing cell " << c << "\nSpecies";
                for (specie_id const& lcfID : this->observed_species_)
                    *csv_out << ',' << specie_str[lcfID];
                
                csv_out->add_div("\navg peak,");
                for (std::size_t s = 0; s < this->observed_species_.size(); ++s)
                    csv_out->add_data(avg_peak[s]);
                
                csv_out->add_div("\navg amp,");
                for (std::size_t s = 0; s < this->observed_species_.size(); ++s)
                    csv_out->add_data(amplitudes[s][c]);
                
                csv_out->add_div("\navg per,");
                for (std::size_t s = 0; s < this->observed_species_.size(); ++s)
                    csv_out->add_data(periods[s][c]);
            }
        }
    }


    Details get_details() override{
        std::vector<Real> times;
        
        
        for(size_t i = 0; i <  peaksAndTroughs.size(); i++){
            for(size_t j = 0; j <  peaksAndTroughs[j].size(); j++){
                for(size_t l = 0; l <  peaksAndTroughs[i][j].size(); l++){
                    times.push_back(peaksAndTroughs[i][j][l].time);
                    detail.concs.push_back( peaksAndTroughs[i][j][l].conc);
                }
            }
        }
        detail.other_details.push_back(times);
        
        return detail;
    }
    
    
    
    //template <typename Simulation>
    void update(Simulation& simulation, std::ostream&) override{
        for (Natural c = this->min; c < this->max; ++c) {
            get_peaks_and_troughs(simulation, c - this->min);
        }
        ++this->samples;
    }


    //template <typename Simulation>
    void finalize() override{
        int timeTemp = this->samples;
        for (std::size_t s = 0; s < this->observed_species_.size(); ++s)
        {
            for (Natural c = 0; c < this->max - this->min; ++c){
                this->samples = timeTemp;
                while (windows[s][c].getSize()>=(range_steps/2)){
                    windows[s][c].dequeue();
                    //bst[s][c].erase(bst[s][c].find(removed));
                    //std::cout<<"bst size="<<bst[s][c].size()<<'\n';
                    checkCritPoint(s, c);
                    ++this->samples;
                }
                calcAmpsAndPers(s, c);
            }
        }
        if(!finalized){
            finalized = true;
        }
    }
};
#endif
