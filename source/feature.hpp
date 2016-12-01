#ifndef FEATURE_HPP
#define FEATURE_HPP



using namespace std;

class feature{
private:
    double period_post; // The period of oscillations for relevant concentrations in the posterior
    double peaktotrough_mid; // The peak to trough ratio for relevant concentrations in the middle of the simulation time-wise
    double peaktotrough_end; // The peak to trough ratio for relevant concentrations at the end of the simulation time-wise
    double sync_score_ant[]; // The synchronization score for the relevant concentrations in the anterior
    double comp_score_ant_mespa; // The score for the complementary expression of her and mespa
    double comp_score_ant_mespb; // The score for the complementary expression of her and mespb
    map<int, double> amplitude_ant_time[]; // The amplitude in the anterior at various time points in the simulaiton
    
public:
    
};

#endif

