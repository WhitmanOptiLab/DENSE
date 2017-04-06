#include "reaction.hpp"

#include <cstddef>

// First, define the principle inputs and outputs for each reaction
// For example, a reaction "one" that consumes nothing but produces 
//   a SPECIE_ONE, you might write:
// 

const int num_inputs_ph1_synthesis = 0;
const int num_outputs_ph1_synthesis = 1;
const int in_counts_ph1_synthesis[] = {};
const specie_id inputs_ph1_synthesis[] = {};
const int out_counts_ph1_synthesis[] = {1};
const specie_id outputs_ph1_synthesis[] = {ph1};
const int num_factors_ph1_synthesis = 1;
const specie_id factors_ph1_synthesis[] = {mh1};

const int num_inputs_ph1_degradation = 1;
const int num_outputs_ph1_degradation = 0;
const int in_counts_ph1_degradation[] = {1};
const specie_id inputs_ph1_degradation[] = {ph1};
const int out_counts_ph1_degradation[] = {};
const specie_id outputs_ph1_degradation[] = {};
const int num_factors_ph1_degradation = 0;
const specie_id factors_ph1_degradation[] = {};

const int num_inputs_ph7_synthesis = 0;
const int num_outputs_ph7_synthesis = 1;
const int in_counts_ph7_synthesis[] = {};
const specie_id inputs_ph7_synthesis[] = {};
const int out_counts_ph7_synthesis[] = {1};
const specie_id outputs_ph7_synthesis[] = {ph7};
const int num_factors_ph7_synthesis = 1;
const specie_id factors_ph7_synthesis[] = {mh7};

const int num_inputs_ph7_degradation = 1;
const int num_outputs_ph7_degradation = 0;
const int in_counts_ph7_degradation[] = {1};
const specie_id inputs_ph7_degradation[] = {ph7};
const int out_counts_ph7_degradation[] = {};
const specie_id outputs_ph7_degradation[] = {};
const int num_factors_ph7_degradation = 0;
const specie_id factors_ph7_degradation[] = {};

const int num_inputs_ph13_synthesis = 0;
const int num_outputs_ph13_synthesis = 1;
const int in_counts_ph13_synthesis[] = {};
const specie_id inputs_ph13_synthesis[] = {};
const int out_counts_ph13_synthesis[] = {1};
const specie_id outputs_ph13_synthesis[] = {ph13};
const int num_factors_ph13_synthesis = 1;
const specie_id factors_ph13_synthesis[] = {mh13};

const int num_inputs_ph13_degradation = 1;
const int num_outputs_ph13_degradation = 0;
const int in_counts_ph13_degradation[] = {1};
const specie_id inputs_ph13_degradation[] = {ph13};
const int out_counts_ph13_degradation[] = {};
const specie_id outputs_ph13_degradation[] = {};
const int num_factors_ph13_degradation = 0;
const specie_id factors_ph13_degradation[] = {};

const int num_inputs_pd_synthesis = 0;
const int num_outputs_pd_synthesis = 1;
const int in_counts_pd_synthesis[] = {};
const specie_id inputs_pd_synthesis[] = {};
const int out_counts_pd_synthesis[] = {1};
const specie_id outputs_pd_synthesis[] = {pd};
const int num_factors_pd_synthesis = 1;
const specie_id factors_pd_synthesis[] = {md};

const int num_inputs_pd_degradation = 1;
const int num_outputs_pd_degradation = 0;
const int in_counts_pd_degradation[] = {1};
const specie_id inputs_pd_degradation[] = {pd};
const int out_counts_pd_degradation[] = {};
const specie_id outputs_pd_degradation[] = {};
const int num_factors_pd_degradation = 0;
const specie_id factors_pd_degradation[] = {};




const int num_inputs_ph11_association = 1;
const int num_outputs_ph11_association = 1;
const int in_counts_ph11_association[] = {2};
const specie_id inputs_ph11_association[] = {ph1};
const int out_counts_ph11_association[] = {1};
const specie_id outputs_ph11_association[] = {ph11};
const int num_factors_ph11_association = 0;
const specie_id factors_ph11_association[] = {};

const int num_inputs_ph77_association = 1;
const int num_outputs_ph77_association = 1;
const int in_counts_ph77_association[] = {2};
const specie_id inputs_ph77_association[] = {ph7};
const int out_counts_ph77_association[] = {1};
const specie_id outputs_ph77_association[] = {ph77};
const int num_factors_ph77_association = 0;
const specie_id factors_ph77_association[] = {};

const int num_inputs_ph1313_association = 1;
const int num_outputs_ph1313_association = 1;
const int in_counts_ph1313_association[] = {2};
const specie_id inputs_ph1313_association[] = {ph13};
const int out_counts_ph1313_association[] = {1};
const specie_id outputs_ph1313_association[] = {ph1313};
const int num_factors_ph1313_association = 0;
const specie_id factors_ph1313_association[] = {};

const int num_inputs_ph17_association = 2;
const int num_outputs_ph17_association = 1;
const int in_counts_ph17_association[] = {1,1};
const specie_id inputs_ph17_association[] = {ph1,ph7};
const int out_counts_ph17_association[] = {1};
const specie_id outputs_ph17_association[] = {ph17};
const int num_factors_ph17_association = 0;
const specie_id factors_ph17_association[] = {};

const int num_inputs_ph113_association = 2;
const int num_outputs_ph113_association = 1;
const int in_counts_ph113_association[] = {1,1};
const specie_id inputs_ph113_association[] = {ph1,ph13};
const int out_counts_ph113_association[] = {1};
const specie_id outputs_ph113_association[] = {ph113};
const int num_factors_ph113_association = 0;
const specie_id factors_ph113_association[] = {};

const int num_inputs_ph713_association = 2;
const int num_outputs_ph713_association = 1;
const int in_counts_ph713_association[] = {1,1};
const specie_id inputs_ph713_association[] = {ph13,ph7};
const int out_counts_ph713_association[] = {1};
const specie_id outputs_ph713_association[] = {ph713};
const int num_factors_ph713_association = 0;
const specie_id factors_ph713_association[] = {};



const int num_inputs_ph11_dissociation = 1;
const int num_outputs_ph11_dissociation = 1;
const int in_counts_ph11_dissociation[] = {1};
const specie_id inputs_ph11_dissociation[] = {ph11};
const int out_counts_ph11_dissociation[] = {2};
const specie_id outputs_ph11_dissociation[] = {ph1};
const int num_factors_ph11_dissociation = 0;
const specie_id factors_ph11_dissociation[] = {};

const int num_inputs_ph77_dissociation = 1;
const int num_outputs_ph77_dissociation = 1;
const int in_counts_ph77_dissociation[] = {1};
const specie_id inputs_ph77_dissociation[] = {ph77};
const int out_counts_ph77_dissociation[] = {2};
const specie_id outputs_ph77_dissociation[] = {ph7};
const int num_factors_ph77_dissociation = 0;
const specie_id factors_ph77_dissociation[] = {};

const int num_inputs_ph1313_dissociation = 1;
const int num_outputs_ph1313_dissociation = 1;
const int in_counts_ph1313_dissociation[] = {1};
const specie_id inputs_ph1313_dissociation[] = {ph1313};
const int out_counts_ph1313_dissociation[] = {2};
const specie_id outputs_ph1313_dissociation[] = {ph13};
const int num_factors_ph1313_dissociation = 0;
const specie_id factors_ph1313_dissociation[] = {};

const int num_inputs_ph17_dissociation = 1;
const int num_outputs_ph17_dissociation = 2;
const int in_counts_ph17_dissociation[] = {1};
const specie_id inputs_ph17_dissociation[] = {ph17};
const int out_counts_ph17_dissociation[] = {1,1};
const specie_id outputs_ph17_dissociation[] = {ph1,ph7};
const int num_factors_ph17_dissociation = 0;
const specie_id factors_ph17_dissociation[] = {};

const int num_inputs_ph113_dissociation = 1;
const int num_outputs_ph113_dissociation = 2;
const int in_counts_ph113_dissociation[] = {1};
const specie_id inputs_ph113_dissociation[] = {ph113};
const int out_counts_ph113_dissociation[] = {1,1};
const specie_id outputs_ph113_dissociation[] = {ph1,ph13};
const int num_factors_ph113_dissociation = 0;
const specie_id factors_ph113_dissociation[] = {};

const int num_inputs_ph713_dissociation = 1;
const int num_outputs_ph713_dissociation = 2;
const int in_counts_ph713_dissociation[] = {1};
const specie_id inputs_ph713_dissociation[] = {ph713};
const int out_counts_ph713_dissociation[] = {1,1};
const specie_id outputs_ph713_dissociation[] = {ph13,ph7};
const int num_factors_ph713_dissociation = 0;
const specie_id factors_ph713_dissociation[] = {};




const int num_inputs_ph11_degradation = 1;
const int num_outputs_ph11_degradation = 0;
const int in_counts_ph11_degradation[] = {1};
const specie_id inputs_ph11_degradation[] = {ph11};
const int out_counts_ph11_degradation[] = {};
const specie_id outputs_ph11_degradation[] = {};
const int num_factors_ph11_degradation = 0;
const specie_id factors_ph11_degradation[] = {};

const int num_inputs_ph77_degradation = 1;
const int num_outputs_ph77_degradation = 0;
const int in_counts_ph77_degradation[] = {1};
const specie_id inputs_ph77_degradation[] = {ph77};
const int out_counts_ph77_degradation[] = {};
const specie_id outputs_ph77_degradation[] = {};
const int num_factors_ph77_degradation = 0;
const specie_id factors_ph77_degradation[] = {};

const int num_inputs_ph1313_degradation = 1;
const int num_outputs_ph1313_degradation = 0;
const int in_counts_ph1313_degradation[] = {1};
const specie_id inputs_ph1313_degradation[] = {ph1313};
const int out_counts_ph1313_degradation[] = {};
const specie_id outputs_ph1313_degradation[] = {};
const int num_factors_ph1313_degradation = 0;
const specie_id factors_ph1313_degradation[] = {};

const int num_inputs_ph17_degradation = 1;
const int num_outputs_ph17_degradation = 0;
const int in_counts_ph17_degradation[] = {1};
const specie_id inputs_ph17_degradation[] = {ph17};
const int out_counts_ph17_degradation[] = {};
const specie_id outputs_ph17_degradation[] = {};
const int num_factors_ph17_degradation = 0;
const specie_id factors_ph17_degradation[] = {};

const int num_inputs_ph113_degradation = 1;
const int num_outputs_ph113_degradation = 0;
const int in_counts_ph113_degradation[] = {1};
const specie_id inputs_ph113_degradation[] = {ph113};
const int out_counts_ph113_degradation[] = {};
const specie_id outputs_ph113_degradation[] = {};
const int num_factors_ph113_degradation = 0;
const specie_id factors_ph113_degradation[] = {};

const int num_inputs_ph713_degradation = 1;
const int num_outputs_ph713_degradation = 0;
const int in_counts_ph713_degradation[] = {1};
const specie_id inputs_ph713_degradation[] = {ph713};
const int out_counts_ph713_degradation[] = {};
const specie_id outputs_ph713_degradation[] = {};
const int num_factors_ph713_degradation = 0;
const specie_id factors_ph713_degradation[] = {};

const int num_inputs_mh1_synthesis = 0;
const int num_outputs_mh1_synthesis = 1;
const int in_counts_mh1_synthesis[] = {};
const specie_id inputs_mh1_synthesis[] = {};
const int out_counts_mh1_synthesis[] = {1};
const specie_id outputs_mh1_synthesis[] = {mh1};
const int num_factors_mh1_synthesis = 3;
const specie_id factors_mh1_synthesis[] = {pd,ph11,ph713};

const int num_inputs_mh1_degradation = 1;
const int num_outputs_mh1_degradation = 0;
const int in_counts_mh1_degradation[] = {1};
const specie_id inputs_mh1_degradation[] = {mh1};
const int out_counts_mh1_degradation[] = {};
const specie_id outputs_mh1_degradation[] = {};
const int num_factors_mh1_degradation = 0;
const specie_id factors_mh1_degradation[] = {};

const int num_inputs_mh7_synthesis = 0;
const int num_outputs_mh7_synthesis = 1;
const int in_counts_mh7_synthesis[] = {};
const specie_id inputs_mh7_synthesis[] = {};
const int out_counts_mh7_synthesis[] = {1};
const specie_id outputs_mh7_synthesis[] = {mh7};
const int num_factors_mh7_synthesis = 3;
const specie_id factors_mh7_synthesis[] = {pd,ph11,ph713};

const int num_inputs_mh7_degradation = 1;
const int num_outputs_mh7_degradation = 0;
const int in_counts_mh7_degradation[] = {1};
const specie_id inputs_mh7_degradation[] = {mh7};
const int out_counts_mh7_degradation[] = {};
const specie_id outputs_mh7_degradation[] = {};
const int num_factors_mh7_degradation = 0;
const specie_id factors_mh7_degradation[] = {};

const int num_inputs_mh13_synthesis = 0;
const int num_outputs_mh13_synthesis = 1;
const int in_counts_mh13_synthesis[] = {};
const specie_id inputs_mh13_synthesis[] = {};
const int out_counts_mh13_synthesis[] = {1};
const specie_id outputs_mh13_synthesis[] = {mh13};
const int num_factors_mh13_synthesis = 0;
const specie_id factors_mh13_synthesis[] = {};

const int num_inputs_mh13_degradation = 1;
const int num_outputs_mh13_degradation = 0;
const int in_counts_mh13_degradation[] = {1};
const specie_id inputs_mh13_degradation[] = {mh13};
const int out_counts_mh13_degradation[] = {};
const specie_id outputs_mh13_degradation[] = {};
const int num_factors_mh13_degradation = 0;
const specie_id factors_mh13_degradation[] = {};

const int num_inputs_md_synthesis = 0;
const int num_outputs_md_synthesis = 1;
const int in_counts_md_synthesis[] = {};
const specie_id inputs_md_synthesis[] = {};
const int out_counts_md_synthesis[] = {1};
const specie_id outputs_md_synthesis[] = {md};
const int num_factors_md_synthesis = 2;
const specie_id factors_md_synthesis[] = {ph11,ph713};

const int num_inputs_md_degradation = 1;
const int num_outputs_md_degradation = 0;
const int in_counts_md_degradation[] = {1};
const specie_id inputs_md_degradation[] = {md};
const int out_counts_md_degradation[] = {};
const specie_id outputs_md_degradation[] = {};
const int num_factors_md_degradation = 0;
const specie_id factors_md_degradation[] = {};






