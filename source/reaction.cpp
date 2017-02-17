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

//dimer dissociation comsumes dimer and produces monomer
//dimer association consumes monomer and produces dimer
const int num_inputs_ph11_dissociation = 1;
const int num_outputs_ph11_dissociation = 2;
const int in_counts_ph11_dissociation[] = {1};
const specie_id inputs_ph11_dissociation[] = {ph11};
const int out_counts_ph11_dissociation[] = {2};
const specie_id outputs_ph11_dissociation[] = {ph1,ph1};
const int num_factors_ph11_dissociation = 0;
const specie_id factors_ph11_dissociation[] = {};

const int num_inputs_ph11_association = 2;
const int num_outputs_ph11_association = 1;
const int in_counts_ph11_association[] = {2};
const specie_id inputs_ph11_association[] = {ph1,ph1};
const int out_counts_ph11_association[] = {1};
const specie_id outputs_ph11_association[] = {ph11};
const int num_factors_ph11_association = 0;
const specie_id factors_ph11_association[] = {};



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



const int num_inputs_pm1_synthesis = 0;
const int num_outputs_pm1_synthesis = 1;
const int in_counts_pm1_synthesis[] = {};
const specie_id inputs_pm1_synthesis[] = {};
const int out_counts_pm1_synthesis[] = {1};
const specie_id outputs_pm1_synthesis[] = {pm1};
const int num_factors_pm1_synthesis = 1;
const specie_id factors_pm1_synthesis[] = {mm1};

const int num_inputs_pm1_degradation = 1;
const int num_outputs_pm1_degradation = 0;
const int in_counts_pm1_degradation[] = {1};
const specie_id inputs_pm1_degradation[] = {pm1};
const int out_counts_pm1_degradation[] = {};
const specie_id outputs_pm1_degradation[] = {};
const int num_factors_pm1_degradation = 0;
const specie_id factors_pm1_degradation[] = {};



const int num_inputs_pm2_synthesis = 0;
const int num_outputs_pm2_synthesis = 1;
const int in_counts_pm2_synthesis[] = {};
const specie_id inputs_pm2_synthesis[] = {};
const int out_counts_pm2_synthesis[] = {1};
const specie_id outputs_pm2_synthesis[] = {pm2};
const int num_factors_pm2_synthesis = 1;
const specie_id factors_pm2_synthesis[] = {mm2};

const int num_inputs_pm2_degradation = 1;
const int num_outputs_pm2_degradation = 0;
const int in_counts_pm2_degradation[] = {1};
const specie_id inputs_pm2_degradation[] = {pm2};
const int out_counts_pm2_degradation[] = {};
const specie_id outputs_pm2_degradation[] = {};
const int num_factors_pm2_degradation = 0;
const specie_id factors_pm2_degradation[] = {};
/*
const int num_inputs_mespb_dissociation = 1;
const int num_outputs_mespb_dissociation = 1;
const int in_counts_mespb_dissociation[] = {1};
const specie_id inputs_mespb_dissociation[] = {pm11};
const int out_counts_mespb_dissociation[] = {1};
const specie_id outputs_mespb_dissociation[] = {pm2};

const int num_inputs_mespb_dissociation2 = 1;
const int num_outputs_mespb_dissociation2 = 1;
const int in_counts_mespb_dissociation2[] = {1};
const specie_id inputs_mespb_dissociation2[] = {pm11};
const int out_counts_mespb_dissociation2[] = {1};
const specie_id outputs_mespb_dissociation2[] = {pm2};

const int num_inputs_mespb_dissociation3 = 1;
const int num_outputs_mespb_dissociation3 = 1;
const int in_counts_mespb_dissociation3[] = {1};
const specie_id inputs_mespb_dissociation3[] = {pm11};
const int out_counts_mespb_dissociation3[] = {1};
const specie_id outputs_mespb_dissociation3[] = {pm2};

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
*/


const int num_inputs_ph11_degradation = 1;
const int num_outputs_ph11_degradation = 0;
const int in_counts_ph11_degradation[] = {1};
const specie_id inputs_ph11_degradation[] = {ph11};
const int out_counts_ph11_degradation[] = {};
const specie_id outputs_ph11_degradation[] = {};
const int num_factors_ph11_degradation = 0;
const specie_id factors_ph11_degradation[] = {};

const int num_inputs_pm11_degradation = 1;
const int num_outputs_pm11_degradation = 0;
const int in_counts_pm11_degradation[] = {1};
const specie_id inputs_pm11_degradation[] = {pm11};
const int out_counts_pm11_degradation[] = {};
const specie_id outputs_pm11_degradation[] = {};
const int num_factors_pm11_degradation = 0;
const specie_id factors_pm11_degradation[] = {};

const int num_inputs_pm12_degradation = 1;
const int num_outputs_pm12_degradation = 0;
const int in_counts_pm12_degradation[] = {1};
const specie_id inputs_pm12_degradation[] = {pm12};
const int out_counts_pm12_degradation[] = {};
const specie_id outputs_pm12_degradation[] = {};
const int num_factors_pm12_degradation = 0;
const specie_id factors_pm12_degradation[] = {};

const int num_inputs_pm22_degradation = 1;
const int num_outputs_pm22_degradation = 0;
const int in_counts_pm22_degradation[] = {1};
const specie_id inputs_pm22_degradation[] = {pm22};
const int out_counts_pm22_degradation[] = {};
const specie_id outputs_pm22_degradation[] = {};
const int num_factors_pm22_degradation = 0;
const specie_id factors_pm22_degradation[] = {};



const int num_inputs_mh1_synthesis = 0;
const int num_outputs_mh1_synthesis = 1;
const int in_counts_mh1_synthesis[] = {};
const specie_id inputs_mh1_synthesis[] = {};
const int out_counts_mh1_synthesis[] = {1};
const specie_id outputs_mh1_synthesis[] = {mh1};
const int num_factors_mh1_synthesis = 2;
const specie_id factors_mh1_synthesis[] = {pd,ph11};

const int num_inputs_mh1_degradation = 1;
const int num_outputs_mh1_degradation = 0;
const int in_counts_mh1_degradation[] = {1};
const specie_id inputs_mh1_degradation[] = {mh1};
const int out_counts_mh1_degradation[] = {};
const specie_id outputs_mh1_degradation[] = {};
const int num_factors_mh1_degradation = 0;
const specie_id factors_mh1_degradation[] = {};


const int num_inputs_md_synthesis = 0;
const int num_outputs_md_synthesis = 1;
const int in_counts_md_synthesis[] = {};
const specie_id inputs_md_synthesis[] = {};
const int out_counts_md_synthesis[] = {1};
const specie_id outputs_md_synthesis[] = {md};
const int num_factors_md_synthesis = 1;
const specie_id factors_md_synthesis[] = {ph11};

const int num_inputs_md_degradation = 1;
const int num_outputs_md_degradation = 0;
const int in_counts_md_degradation[] = {1};
const specie_id inputs_md_degradation[] = {md};
const int out_counts_md_degradation[] = {};
const specie_id outputs_md_degradation[] = {};
const int num_factors_md_degradation = 0;
const specie_id factors_md_degradation[] = {};


const int num_inputs_mm1_synthesis = 0;
const int num_outputs_mm1_synthesis = 1;
const int in_counts_mm1_synthesis[] = {};
const specie_id inputs_mm1_synthesis[] = {};
const int out_counts_mm1_synthesis[] = {1};
const specie_id outputs_mm1_synthesis[] = {mm1};
const int num_factors_mm1_synthesis = 2;
const specie_id factors_mm1_synthesis[] = {pd, ph11};

const int num_inputs_mm1_degradation = 1;
const int num_outputs_mm1_degradation = 0;
const int in_counts_mm1_degradation[] = {1};
const specie_id inputs_mm1_degradation[] = {mm1};
const int out_counts_mm1_degradation[] = {};
const specie_id outputs_mm1_degradation[] = {};
const int num_factors_mm1_degradation = 0;
const specie_id factors_mm1_degradation[] = {};


const int num_inputs_mm2_synthesis = 0;
const int num_outputs_mm2_synthesis = 1;
const int in_counts_mm2_synthesis[] = {};
const specie_id inputs_mm2_synthesis[] = {};
const int out_counts_mm2_synthesis[] = {1};
const specie_id outputs_mm2_synthesis[] = {mm2};
const int num_factors_mm2_synthesis = 3;
const specie_id factors_mm2_synthesis[] = {pd,pm11,pm22};

const int num_inputs_mm2_degradation = 1;
const int num_outputs_mm2_degradation = 0;
const int in_counts_mm2_degradation[] = {1};
const specie_id inputs_mm2_degradation[] = {mm2};
const int out_counts_mm2_degradation[] = {};
const specie_id outputs_mm2_degradation[] = {};
const int num_factors_mm2_degradation = 0;
const specie_id factors_mm2_degradation[] = {};


const int num_inputs_pm11_dissociation = 1;
const int num_outputs_pm11_dissociation = 1;
const int in_counts_pm11_dissociation[] = {1};
const specie_id inputs_pm11_dissociation[] = {pm11};
const int out_counts_pm11_dissociation[] = {1};
const specie_id outputs_pm11_dissociation[] = {pm1};
const int num_factors_pm11_dissociation = 0;
const specie_id factors_pm11_dissociation[] = {};

const int num_inputs_pm12_dissociation = 1;
const int num_outputs_pm12_dissociation = 2;
const int in_counts_pm12_dissociation[] = {1};
const specie_id inputs_pm12_dissociation[] = {pm12};
const int out_counts_pm12_dissociation[] = {2};
const specie_id outputs_pm12_dissociation[] = {pm1,pm2};
const int num_factors_pm12_dissociation = 0;
const specie_id factors_pm12_dissociation[] = {};

const int num_inputs_pm22_dissociation = 1;
const int num_outputs_pm22_dissociation = 1;
const int in_counts_pm22_dissociation[] = {1};
const specie_id inputs_pm22_dissociation[] = {pm22};
const int out_counts_pm22_dissociation[] = {1};
const specie_id outputs_pm22_dissociation[] = {pm2};
const int num_factors_pm22_dissociation = 0;
const specie_id factors_pm22_dissociation[] = {};

const int num_inputs_pm11_association = 1;
const int num_outputs_pm11_association = 1;
const int in_counts_pm11_association[] = {1};
const specie_id inputs_pm11_association[] = {pm1};
const int out_counts_pm11_association[] = {1};
const specie_id outputs_pm11_association[] = {pm11};
const int num_factors_pm11_association = 0;
const specie_id factors_pm11_association[] = {};

const int num_inputs_pm12_association = 2;
const int num_outputs_pm12_association = 1;
const int in_counts_pm12_association[] = {2};
const specie_id inputs_pm12_association[] = {pm1,pm2};
const int out_counts_pm12_association[] = {1};
const specie_id outputs_pm12_association[] = {pm12};
const int num_factors_pm12_association = 0;
const specie_id factors_pm12_association[] = {};

const int num_inputs_pm22_association = 1;
const int num_outputs_pm22_association = 1;
const int in_counts_pm22_association[] = {1};
const specie_id inputs_pm22_association[] = {pm2};
const int out_counts_pm22_association[] = {1};
const specie_id outputs_pm22_association[] = {pm22};
const int num_factors_pm22_association = 0;
const specie_id factors_pm22_association[] = {};
