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

const int num_inputs_ph1_degradation = 0;
const int num_outputs_ph1_degradation = 1;
const int in_counts_ph1_degradation[] = {};
const specie_id inputs_ph1_degradation[] = {};
const int out_counts_ph1_degradation[] = {1};
const specie_id outputs_ph1_degradation[] = {ph1};

const int num_inputs_ph1_dissociation = 0;
const int num_outputs_ph1_dissociation = 2;
const int in_counts_ph1_dissociation[] = {};
const specie_id inputs_ph1_dissociation[] = {};
const int out_counts_ph1_dissociation[] = {2};
const specie_id outputs_ph1_dissociation[] = {ph1,ph11};

const int num_inputs_ph1_association = 0;
const int num_outputs_ph1_association = 2;
const int in_counts_ph1_association[] = {};
const specie_id inputs_ph1_association[] = {};
const int out_counts_ph1_association[] = {2};
const specie_id outputs_ph1_association[] = {ph1,ph11};



const int num_inputs_pd_synthesis = 0;
const int num_outputs_pd_synthesis = 1;
const int in_counts_pd_synthesis[] = {};
const specie_id inputs_pd_synthesis[] = {};
const int out_counts_pd_synthesis[] = {1};
const specie_id outputs_pd_synthesis[] = {pd};

const int num_inputs_pd_degradation = 0;
const int num_outputs_pd_degradation = 1;
const int in_counts_pd_degradation[] = {};
const specie_id inputs_pd_degradation[] = {};
const int out_counts_pd_degradation[] = {1};
const specie_id outputs_pd_degradation[] = {pd};



const int num_inputs_mespa_synthesis = 0;
const int num_outputs_mespa_synthesis = 1;
const int in_counts_mespa_synthesis[] = {};
const specie_id inputs_mespa_synthesis[] = {};
const int out_counts_mespa_synthesis[] = {1};
const specie_id outputs_mespa_synthesis[] = {pm1};

const int num_inputs_mepsa_degradation = 0;
const int num_outputs_mepsa_degradation = 1;
const int in_counts_mepsa_degradation[] = {};
const specie_id inputs_mepsa_degradation[] = {};
const int out_counts_mepsa_degradation[] = {1};
const specie_id outputs_mepsa_degradation[] = {pm1};

const int num_inputs_mespa_dissociation = 0;
const int num_outputs_mespa_dissociation = 2;
const int in_counts_mespa_dissociation[] = {};
const specie_id inputs_mespa_dissociation[] = {};
const int out_counts_mespa_dissociation[] = {2};
const specie_id outputs_mespa_dissociation[] = {pm1,pm11};

const int num_inputs_mespa_dissociation2 = 0;
const int num_outputs_mespa_dissociation2 = 2;
const int in_counts_mespa_dissociation2[] = {};
const specie_id inputs_mespa_dissociation2[] = {};
const int out_counts_mespa_dissociation2[] = {2};
const specie_id outputs_mespa_dissociation2[] = {pm1,pm12};

const int num_inputs_mespa_dissociation3 = 0;
const int num_outputs_mespa_dissociation3 = 2;
const int in_counts_mespa_dissociation3[] = {};
const specie_id inputs_mespa_dissociation3[] = {};
const int out_counts_mespa_dissociation3[] = {2};
const specie_id outputs_mespa_dissociation3[] = {pm1,pm22};

const int num_inputs_mespa_association = 0;
const int num_outputs_mespa_association = 2;
const int in_counts_mespa_association[] = {};
const specie_id inputs_mespa_association[] = {};
const int out_counts_mespa_association[] = {2};
const specie_id outputs_mespa_association[] = {pm1,pm11};

const int num_inputs_mespa_association2 = 0;
const int num_outputs_mespa_association2 = 2;
const int in_counts_mespa_association2[] = {};
const specie_id inputs_mespa_association2[] = {};
const int out_counts_mespa_association2[] = {2};
const specie_id outputs_mespa_association2[] = {pm1,pm12};

const int num_inputs_mespa_association3 = 0;
const int num_outputs_mespa_association3 = 2;
const int in_counts_mespa_association3[] = {};
const specie_id inputs_mespa_association3[] = {};
const int out_counts_mespa_association3[] = {2};
const specie_id outputs_mespa_association3[] = {pm1,pm22};



const int num_inputs_mespb_synthesis = 0;
const int num_outputs_mespb_synthesis = 1;
const int in_counts_mespb_synthesis[] = {};
const specie_id inputs_mespb_synthesis[] = {};
const int out_counts_mespb_synthesis[] = {1};
const specie_id outputs_mespb_synthesis[] = {pm2};

const int num_inputs_mepsb_degradation = 0;
const int num_outputs_mepsb_degradation = 1;
const int in_counts_mepsb_degradation[] = {};
const specie_id inputs_mepsb_degradation[] = {};
const int out_counts_mepsb_degradation[] = {1};
const specie_id outputs_mepsb_degradation[] = {pm2};

const int num_inputs_mespb_dissociation = 0;
const int num_outputs_mespb_dissociation = 2;
const int in_counts_mespb_dissociation[] = {};
const specie_id inputs_mespb_dissociation[] = {};
const int out_counts_mespb_dissociation[] = {2};
const specie_id outputs_mespb_dissociation[] = {pm2,pm11};

const int num_inputs_mespb_dissociation2 = 0;
const int num_outputs_mespb_dissociation2 = 2;
const int in_counts_mespb_dissociation2[] = {};
const specie_id inputs_mespb_dissociation2[] = {};
const int out_counts_mespb_dissociation2[] = {2};
const specie_id outputs_mespb_dissociation2[] = {pm2,pm12};

const int num_inputs_mespb_dissociation3 = 0;
const int num_outputs_mespb_dissociation3 = 2;
const int in_counts_mespb_dissociation3[] = {};
const specie_id inputs_mespb_dissociation3[] = {};
const int out_counts_mespb_dissociation3[] = {2};
const specie_id outputs_mespb_dissociation3[] = {pm2,pm22};

const int num_inputs_mespb_association = 0;
const int num_outputs_mespb_association = 2;
const int in_counts_mespb_association[] = {};
const specie_id inputs_mespb_association[] = {};
const int out_counts_mespb_association[] = {2};
const specie_id outputs_mespb_association[] = {pm2,pm11};

const int num_inputs_mespb_association2 = 0;
const int num_outputs_mespb_association2 = 2;
const int in_counts_mespb_association2[] = {};
const specie_id inputs_mespb_association2[] = {};
const int out_counts_mespb_association2[] = {2};
const specie_id outputs_mespb_association2[] = {pm2,pm12};

const int num_inputs_mespb_association3 = 0;
const int num_outputs_mespb_association3 = 2;
const int in_counts_mespb_association3[] = {};
const specie_id inputs_mespb_association3[] = {};
const int out_counts_mespb_association3[] = {2};
const specie_id outputs_mespb_association3[] = {pm2,pm22};



const int num_inputs_ph11_degradation = 0;
const int num_outputs_ph11_degradation = 1;
const int in_counts_ph11_degradation[] = {};
const specie_id inputs_ph11_degradation[] = {};
const int out_counts_ph11_degradation[] = {1};
const specie_id outputs_ph11_degradation[] = {ph11};

const int num_inputs_pm11_degradation = 0;
const int num_outputs_pm11_degradation = 1;
const int in_counts_pm11_degradation[] = {};
const specie_id inputs_pm11_degradation[] = {};
const int out_counts_pm11_degradation[] = {1};
const specie_id outputs_pm11_degradation[] = {pm11};

const int num_inputs_pm12_degradation = 0;
const int num_outputs_pm12_degradation = 1;
const int in_counts_pm12_degradation[] = {};
const specie_id inputs_pm12_degradation[] = {};
const int out_counts_pm12_degradation[] = {1};
const specie_id outputs_pm12_degradation[] = {pm12};

const int num_inputs_pm22_degradation = 0;
const int num_outputs_pm22_degradation = 1;
const int in_counts_pm22_degradation[] = {};
const specie_id inputs_pm22_degradation[] = {};
const int out_counts_pm22_degradation[] = {1};
const specie_id outputs_pm22_degradation[] = {pm22};



const int num_inputs_mh1_synthesis = 0;
const int num_outputs_mh1_synthesis = 1;
const int in_counts_mh1_synthesis[] = {};
const specie_id inputs_mh1_synthesis[] = {};
const int out_counts_mh1_synthesis[] = {1};
const specie_id outputs_mh1_synthesis[] = {mh1};

const int num_inputs_mh1_degradation = 0;
const int num_outputs_mh1_degradation = 1;
const int in_counts_mh1_degradation[] = {};
const specie_id inputs_mh1_degradation[] = {};
const int out_counts_mh1_degradation[] = {1};
const specie_id outputs_mh1_degradation[] = {mh1};


const int num_inputs_md_synthesis = 0;
const int num_outputs_md_synthesis = 1;
const int in_counts_md_synthesis[] = {};
const specie_id inputs_md_synthesis[] = {};
const int out_counts_md_synthesis[] = {1};
const specie_id outputs_md_synthesis[] = {md};

const int num_inputs_md_degradation = 0;
const int num_outputs_md_degradation = 1;
const int in_counts_md_degradation[] = {};
const specie_id inputs_md_degradation[] = {};
const int out_counts_md_degradation[] = {1};
const specie_id outputs_md_degradation[] = {md};


const int num_inputs_mm1_synthesis = 0;
const int num_outputs_mm1_synthesis = 1;
const int in_counts_mm1_synthesis[] = {};
const specie_id inputs_mm1_synthesis[] = {};
const int out_counts_mm1_synthesis[] = {1};
const specie_id outputs_mm1_synthesis[] = {mm1};

const int num_inputs_mm1_degradation = 0;
const int num_outputs_mm1_degradation = 1;
const int in_counts_mm1_degradation[] = {};
const specie_id inputs_mm1_degradation[] = {};
const int out_counts_mm1_degradation[] = {1};
const specie_id outputs_mm1_degradation[] = {mm1};


const int num_inputs_mm2_synthesis = 0;
const int num_outputs_mm2_synthesis = 1;
const int in_counts_mm2_synthesis[] = {};
const specie_id inputs_mm2_synthesis[] = {};
const int out_counts_mm2_synthesis[] = {1};
const specie_id outputs_mm2_synthesis[] = {mm2};

const int num_inputs_mm2_degradation = 0;
const int num_outputs_mm2_degradation = 1;
const int in_counts_mm2_degradation[] = {};
const specie_id inputs_mm2_degradation[] = {};
const int out_counts_mm2_degradation[] = {1};
const specie_id outputs_mm2_degradation[] = {mm2};


