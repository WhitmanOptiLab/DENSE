#include "core/reaction.hpp"

#include <cstddef>

// First, define the principle inputs and outputs for each reaction
// For example, a reaction "one" that consumes nothing but produces 
//   a SPECIE_ONE, you might write:
// 

STATIC_VAR int num_deltas_ph1_synthesis = 1;
STATIC_VAR int deltas_ph1_synthesis[] = {1};
STATIC_VAR specie_id delta_ids_ph1_synthesis[] = {ph1};

STATIC_VAR int num_deltas_ph1_degradation = 1;
STATIC_VAR int deltas_ph1_degradation[] = {-1};
STATIC_VAR specie_id delta_ids_ph1_degradation[] = {ph1};

STATIC_VAR int num_deltas_ph7_synthesis = 1;
STATIC_VAR int deltas_ph7_synthesis[] = {1};
STATIC_VAR specie_id delta_ids_ph7_synthesis[] = {ph7};

STATIC_VAR int num_deltas_ph7_degradation = 1;
STATIC_VAR int deltas_ph7_degradation[] = {-1};
STATIC_VAR specie_id delta_ids_ph7_degradation[] = {ph7};

STATIC_VAR int num_deltas_ph13_synthesis = 1;
STATIC_VAR int deltas_ph13_synthesis[] = {1};
STATIC_VAR specie_id delta_ids_ph13_synthesis[] = {ph13};

STATIC_VAR int num_deltas_ph13_degradation = 1;
STATIC_VAR int deltas_ph13_degradation[] = {-1};
STATIC_VAR specie_id delta_ids_ph13_degradation[] = {ph13};

STATIC_VAR int num_deltas_pd_synthesis = 1;
STATIC_VAR int deltas_pd_synthesis[] = {1};
STATIC_VAR specie_id delta_ids_pd_synthesis[] = {pd};

STATIC_VAR int num_deltas_pd_degradation = 1;
STATIC_VAR int deltas_pd_degradation[] = {-1};
STATIC_VAR specie_id delta_ids_pd_degradation[] = {pd};

STATIC_VAR int num_deltas_ph11_association = 2;
STATIC_VAR int deltas_ph11_association[] = {-2,1};
STATIC_VAR specie_id delta_ids_ph11_association[] = {ph1,ph11};

STATIC_VAR int num_deltas_ph77_association = 2;
STATIC_VAR int deltas_ph77_association[] = {-2,1};
STATIC_VAR specie_id delta_ids_ph77_association[] = {ph7,ph77};

STATIC_VAR int num_deltas_ph1313_association = 2;
STATIC_VAR int deltas_ph1313_association[] = {-2,1};
STATIC_VAR specie_id delta_ids_ph1313_association[] = {ph13,ph1313};

STATIC_VAR int num_deltas_ph17_association = 3;
STATIC_VAR int deltas_ph17_association[] = {-1,-1,1};
STATIC_VAR specie_id delta_ids_ph17_association[] = {ph1,ph7,ph17};

STATIC_VAR int num_deltas_ph113_association = 3;
STATIC_VAR int deltas_ph113_association[] = {-1,-1,1};
STATIC_VAR specie_id delta_ids_ph113_association[] = {ph1,ph13,ph113};

STATIC_VAR int num_deltas_ph713_association = 3;
STATIC_VAR int deltas_ph713_association[] = {-1,-1,1};
STATIC_VAR specie_id delta_ids_ph713_association[] = {ph7,ph13,ph713};

STATIC_VAR int num_deltas_ph11_dissociation = 2;
STATIC_VAR int deltas_ph11_dissociation[] = {-1,2};
STATIC_VAR specie_id delta_ids_ph11_dissociation[] = {ph11,ph1};

STATIC_VAR int num_deltas_ph77_dissociation = 2;
STATIC_VAR int deltas_ph77_dissociation[] = {-1,2};
STATIC_VAR specie_id delta_ids_ph77_dissociation[] = {ph77,ph7};

STATIC_VAR int num_deltas_ph1313_dissociation = 2;
STATIC_VAR int deltas_ph1313_dissociation[] = {-1,2};
STATIC_VAR specie_id delta_ids_ph1313_dissociation[] = {ph1313,ph13};

STATIC_VAR int num_deltas_ph17_dissociation = 3;
STATIC_VAR int deltas_ph17_dissociation[] = {-1,1,1};
STATIC_VAR specie_id delta_ids_ph17_dissociation[] = {ph17,ph1,ph7};

STATIC_VAR int num_deltas_ph113_dissociation = 3;
STATIC_VAR int deltas_ph113_dissociation[] = {-1,1,1};
STATIC_VAR specie_id delta_ids_ph113_dissociation[] = {ph113,ph1,ph13};

STATIC_VAR int num_deltas_ph713_dissociation = 3;
STATIC_VAR int deltas_ph713_dissociation[] = {-1,1,1};
STATIC_VAR specie_id delta_ids_ph713_dissociation[] = {ph713,ph7,ph13};

STATIC_VAR int num_deltas_ph11_degradation = 1;
STATIC_VAR int deltas_ph11_degradation[] = {-1};
STATIC_VAR specie_id delta_ids_ph11_degradation[] = {ph11};

STATIC_VAR int num_deltas_ph77_degradation = 1;
STATIC_VAR int deltas_ph77_degradation[] = {-1};
STATIC_VAR specie_id delta_ids_ph77_degradation[] = {ph77};

STATIC_VAR int num_deltas_ph1313_degradation = 1;
STATIC_VAR int deltas_ph1313_degradation[] = {-1};
STATIC_VAR specie_id delta_ids_ph1313_degradation[] = {ph1313};

STATIC_VAR int num_deltas_ph17_degradation = 1;
STATIC_VAR int deltas_ph17_degradation[] = {-1};
STATIC_VAR specie_id delta_ids_ph17_degradation[] = {ph17};

STATIC_VAR int num_deltas_ph113_degradation = 1;
STATIC_VAR int deltas_ph113_degradation[] = {-1};
STATIC_VAR specie_id delta_ids_ph113_degradation[] = {ph113};

STATIC_VAR int num_deltas_ph713_degradation = 1;
STATIC_VAR int deltas_ph713_degradation[] = {-1};
STATIC_VAR specie_id delta_ids_ph713_degradation[] = {ph713};

STATIC_VAR int num_deltas_mh1_synthesis = 1;
STATIC_VAR int deltas_mh1_synthesis[] = {1};
STATIC_VAR specie_id delta_ids_mh1_synthesis[] = {mh1};

STATIC_VAR int num_deltas_mh1_degradation = 1;
STATIC_VAR int deltas_mh1_degradation[] = {-1};
STATIC_VAR specie_id delta_ids_mh1_degradation[] = {mh1};

STATIC_VAR int num_deltas_mh7_synthesis = 1;
STATIC_VAR int deltas_mh7_synthesis[] = {1};
STATIC_VAR specie_id delta_ids_mh7_synthesis[] = {mh7};

STATIC_VAR int num_deltas_mh7_degradation = 1;
STATIC_VAR int deltas_mh7_degradation[] = {-1};
STATIC_VAR specie_id delta_ids_mh7_degradation[] = {mh7};

STATIC_VAR int num_deltas_mh13_synthesis = 1;
STATIC_VAR int deltas_mh13_synthesis[] = {1};
STATIC_VAR specie_id delta_ids_mh13_synthesis[] = {mh13};

STATIC_VAR int num_deltas_mh13_degradation = 1;
STATIC_VAR int deltas_mh13_degradation[] = {-1};
STATIC_VAR specie_id delta_ids_mh13_degradation[] = {mh13};

STATIC_VAR int num_deltas_md_synthesis = 1;
STATIC_VAR int deltas_md_synthesis[] = {1};
STATIC_VAR specie_id delta_ids_md_synthesis[] = {md};

STATIC_VAR int num_deltas_md_degradation = 1;
STATIC_VAR int deltas_md_degradation[] = {-1};
STATIC_VAR specie_id delta_ids_md_degradation[] = {md};



