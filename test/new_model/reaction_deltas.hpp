#include "core/reaction.hpp"

#include <cstddef>

// First, define the principle inputs and outputs for each reaction
// For example, a reaction "one" that consumes nothing but produces 
//   a SPECIE_ONE, you might write:
// 

STATIC_VAR int num_inputs_ph1_synthesis = 0;
STATIC_VAR int num_outputs_ph1_synthesis = 1;
STATIC_VAR int in_counts_ph1_synthesis[] = {};
STATIC_VAR specie_id inputs_ph1_synthesis[] = {};
STATIC_VAR int out_counts_ph1_synthesis[] = {1};
STATIC_VAR specie_id outputs_ph1_synthesis[] = {ph1};
STATIC_VAR int num_factors_ph1_synthesis = 1;
STATIC_VAR specie_id factors_ph1_synthesis[] = {mh1};

STATIC_VAR int num_inputs_ph7_synthesis = 0;
STATIC_VAR int num_outputs_ph7_synthesis = 1;
STATIC_VAR int in_counts_ph7_synthesis[] = {};
STATIC_VAR specie_id inputs_ph7_synthesis[] = {};
STATIC_VAR int out_counts_ph7_synthesis[] = {1};
STATIC_VAR specie_id outputs_ph7_synthesis[] = {ph7};
STATIC_VAR int num_factors_ph7_synthesis = 1;
STATIC_VAR specie_id factors_ph7_synthesis[] = {mh7};

STATIC_VAR int num_inputs_ph13_synthesis = 0;
STATIC_VAR int num_outputs_ph13_synthesis = 1;
STATIC_VAR int in_counts_ph13_synthesis[] = {};
STATIC_VAR specie_id inputs_ph13_synthesis[] = {};
STATIC_VAR int out_counts_ph13_synthesis[] = {1};
STATIC_VAR specie_id outputs_ph13_synthesis[] = {ph13};
STATIC_VAR int num_factors_ph13_synthesis = 1;
STATIC_VAR specie_id factors_ph13_synthesis[] = {mh13};

STATIC_VAR int num_inputs_pd_synthesis = 0;
STATIC_VAR int num_outputs_pd_synthesis = 1;
STATIC_VAR int in_counts_pd_synthesis[] = {};
STATIC_VAR specie_id inputs_pd_synthesis[] = {};
STATIC_VAR int out_counts_pd_synthesis[] = {1};
STATIC_VAR specie_id outputs_pd_synthesis[] = {pd};
STATIC_VAR int num_factors_pd_synthesis = 1;
STATIC_VAR specie_id factors_pd_synthesis[] = {md};




STATIC_VAR int num_inputs_ph11_association = 1;
STATIC_VAR int num_outputs_ph11_association = 1;
STATIC_VAR int in_counts_ph11_association[] = {2};
STATIC_VAR specie_id inputs_ph11_association[] = {ph1};
STATIC_VAR int out_counts_ph11_association[] = {1};
STATIC_VAR specie_id outputs_ph11_association[] = {ph11};
STATIC_VAR int num_factors_ph11_association = 0;
STATIC_VAR specie_id factors_ph11_association[] = {};

STATIC_VAR int num_inputs_ph77_association = 1;
STATIC_VAR int num_outputs_ph77_association = 1;
STATIC_VAR int in_counts_ph77_association[] = {2};
STATIC_VAR specie_id inputs_ph77_association[] = {ph7};
STATIC_VAR int out_counts_ph77_association[] = {1};
STATIC_VAR specie_id outputs_ph77_association[] = {ph77};
STATIC_VAR int num_factors_ph77_association = 0;
STATIC_VAR specie_id factors_ph77_association[] = {};

STATIC_VAR int num_inputs_ph1313_association = 1;
STATIC_VAR int num_outputs_ph1313_association = 1;
STATIC_VAR int in_counts_ph1313_association[] = {2};
STATIC_VAR specie_id inputs_ph1313_association[] = {ph13};
STATIC_VAR int out_counts_ph1313_association[] = {1};
STATIC_VAR specie_id outputs_ph1313_association[] = {ph1313};
STATIC_VAR int num_factors_ph1313_association = 0;
STATIC_VAR specie_id factors_ph1313_association[] = {};

STATIC_VAR int num_inputs_ph17_association = 2;
STATIC_VAR int num_outputs_ph17_association = 1;
STATIC_VAR int in_counts_ph17_association[] = {1,1};
STATIC_VAR specie_id inputs_ph17_association[] = {ph1,ph7};
STATIC_VAR int out_counts_ph17_association[] = {1};
STATIC_VAR specie_id outputs_ph17_association[] = {ph17};
STATIC_VAR int num_factors_ph17_association = 0;
STATIC_VAR specie_id factors_ph17_association[] = {};

STATIC_VAR int num_inputs_ph113_association = 2;
STATIC_VAR int num_outputs_ph113_association = 1;
STATIC_VAR int in_counts_ph113_association[] = {1,1};
STATIC_VAR specie_id inputs_ph113_association[] = {ph1,ph13};
STATIC_VAR int out_counts_ph113_association[] = {1};
STATIC_VAR specie_id outputs_ph113_association[] = {ph113};
STATIC_VAR int num_factors_ph113_association = 0;
STATIC_VAR specie_id factors_ph113_association[] = {};

STATIC_VAR int num_inputs_ph713_association = 2;
STATIC_VAR int num_outputs_ph713_association = 1;
STATIC_VAR int in_counts_ph713_association[] = {1,1};
STATIC_VAR specie_id inputs_ph713_association[] = {ph13,ph7};
STATIC_VAR int out_counts_ph713_association[] = {1};
STATIC_VAR specie_id outputs_ph713_association[] = {ph713};
STATIC_VAR int num_factors_ph713_association = 0;
STATIC_VAR specie_id factors_ph713_association[] = {};



STATIC_VAR int num_inputs_ph11_dissociation = 1;
STATIC_VAR int num_outputs_ph11_dissociation = 1;
STATIC_VAR int in_counts_ph11_dissociation[] = {1};
STATIC_VAR specie_id inputs_ph11_dissociation[] = {ph11};
STATIC_VAR int out_counts_ph11_dissociation[] = {2};
STATIC_VAR specie_id outputs_ph11_dissociation[] = {ph1};
STATIC_VAR int num_factors_ph11_dissociation = 0;
STATIC_VAR specie_id factors_ph11_dissociation[] = {};

STATIC_VAR int num_inputs_ph77_dissociation = 1;
STATIC_VAR int num_outputs_ph77_dissociation = 1;
STATIC_VAR int in_counts_ph77_dissociation[] = {1};
STATIC_VAR specie_id inputs_ph77_dissociation[] = {ph77};
STATIC_VAR int out_counts_ph77_dissociation[] = {2};
STATIC_VAR specie_id outputs_ph77_dissociation[] = {ph7};
STATIC_VAR int num_factors_ph77_dissociation = 0;
STATIC_VAR specie_id factors_ph77_dissociation[] = {};

STATIC_VAR int num_inputs_ph1313_dissociation = 1;
STATIC_VAR int num_outputs_ph1313_dissociation = 1;
STATIC_VAR int in_counts_ph1313_dissociation[] = {1};
STATIC_VAR specie_id inputs_ph1313_dissociation[] = {ph1313};
STATIC_VAR int out_counts_ph1313_dissociation[] = {2};
STATIC_VAR specie_id outputs_ph1313_dissociation[] = {ph13};
STATIC_VAR int num_factors_ph1313_dissociation = 0;
STATIC_VAR specie_id factors_ph1313_dissociation[] = {};

STATIC_VAR int num_inputs_ph17_dissociation = 1;
STATIC_VAR int num_outputs_ph17_dissociation = 2;
STATIC_VAR int in_counts_ph17_dissociation[] = {1};
STATIC_VAR specie_id inputs_ph17_dissociation[] = {ph17};
STATIC_VAR int out_counts_ph17_dissociation[] = {1,1};
STATIC_VAR specie_id outputs_ph17_dissociation[] = {ph1,ph7};
STATIC_VAR int num_factors_ph17_dissociation = 0;
STATIC_VAR specie_id factors_ph17_dissociation[] = {};

STATIC_VAR int num_inputs_ph113_dissociation = 1;
STATIC_VAR int num_outputs_ph113_dissociation = 2;
STATIC_VAR int in_counts_ph113_dissociation[] = {1};
STATIC_VAR specie_id inputs_ph113_dissociation[] = {ph113};
STATIC_VAR int out_counts_ph113_dissociation[] = {1,1};
STATIC_VAR specie_id outputs_ph113_dissociation[] = {ph1,ph13};
STATIC_VAR int num_factors_ph113_dissociation = 0;
STATIC_VAR specie_id factors_ph113_dissociation[] = {};

STATIC_VAR int num_inputs_ph713_dissociation = 1;
STATIC_VAR int num_outputs_ph713_dissociation = 2;
STATIC_VAR int in_counts_ph713_dissociation[] = {1};
STATIC_VAR specie_id inputs_ph713_dissociation[] = {ph713};
STATIC_VAR int out_counts_ph713_dissociation[] = {1,1};
STATIC_VAR specie_id outputs_ph713_dissociation[] = {ph13,ph7};
STATIC_VAR int num_factors_ph713_dissociation = 0;
STATIC_VAR specie_id factors_ph713_dissociation[] = {};




STATIC_VAR int num_inputs_mh1_synthesis = 0;
STATIC_VAR int num_outputs_mh1_synthesis = 1;
STATIC_VAR int in_counts_mh1_synthesis[] = {};
STATIC_VAR specie_id inputs_mh1_synthesis[] = {};
STATIC_VAR int out_counts_mh1_synthesis[] = {1};
STATIC_VAR specie_id outputs_mh1_synthesis[] = {mh1};
STATIC_VAR int num_factors_mh1_synthesis = 3;
STATIC_VAR specie_id factors_mh1_synthesis[] = {pd,ph11,ph713};

STATIC_VAR int num_inputs_mh7_synthesis = 0;
STATIC_VAR int num_outputs_mh7_synthesis = 1;
STATIC_VAR int in_counts_mh7_synthesis[] = {};
STATIC_VAR specie_id inputs_mh7_synthesis[] = {};
STATIC_VAR int out_counts_mh7_synthesis[] = {1};
STATIC_VAR specie_id outputs_mh7_synthesis[] = {mh7};
STATIC_VAR int num_factors_mh7_synthesis = 3;
STATIC_VAR specie_id factors_mh7_synthesis[] = {pd,ph11,ph713};

STATIC_VAR int num_inputs_mh13_synthesis = 0;
STATIC_VAR int num_outputs_mh13_synthesis = 1;
STATIC_VAR int in_counts_mh13_synthesis[] = {};
STATIC_VAR specie_id inputs_mh13_synthesis[] = {};
STATIC_VAR int out_counts_mh13_synthesis[] = {1};
STATIC_VAR specie_id outputs_mh13_synthesis[] = {mh13};
STATIC_VAR int num_factors_mh13_synthesis = 0;
STATIC_VAR specie_id factors_mh13_synthesis[] = {};

STATIC_VAR int num_inputs_md_synthesis = 0;
STATIC_VAR int num_outputs_md_synthesis = 1;
STATIC_VAR int in_counts_md_synthesis[] = {};
STATIC_VAR specie_id inputs_md_synthesis[] = {};
STATIC_VAR int out_counts_md_synthesis[] = {1};
STATIC_VAR specie_id outputs_md_synthesis[] = {md};
STATIC_VAR int num_factors_md_synthesis = 2;
STATIC_VAR specie_id factors_md_synthesis[] = {ph11,ph713};






