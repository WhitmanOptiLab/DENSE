#include "reaction.hpp"

#include <cstddef>

// First, define the principle inputs and outputs for each reaction
// For example, a reaction "one" that consumes nothing but produces 
//   a SPECIE_ONE, you might write:
// 

STATIC_VAR int num_inputs_mRNA_synthesis = 0;
STATIC_VAR int num_outputs_mRNA_synthesis = 1;
STATIC_VAR int in_counts_mRNA_synthesis[] = {};
STATIC_VAR specie_id inputs_mRNA_synthesis[] = {};
STATIC_VAR int out_counts_mRNA_synthesis[] = {1};
STATIC_VAR specie_id outputs_mRNA_synthesis[] = {mRNA};
STATIC_VAR int num_factors_mRNA_synthesis = 1;
STATIC_VAR specie_id factors_mRNA_synthesis[] = {protein};

STATIC_VAR int num_inputs_protein_synthesis = 0;
STATIC_VAR int num_outputs_protein_synthesis = 1;
STATIC_VAR int in_counts_protein_synthesis[] = {};
STATIC_VAR specie_id inputs_protein_synthesis[] = {};
STATIC_VAR int out_counts_protein_synthesis[] = {1};
STATIC_VAR specie_id outputs_protein_synthesis[] = {protein};
STATIC_VAR int num_factors_protein_synthesis = 1;
STATIC_VAR specie_id factors_protein_synthesis[] = {mRNA};


STATIC_VAR int num_inputs_mRNA2_synthesis = 0;
STATIC_VAR int num_outputs_mRNA2_synthesis = 1;
STATIC_VAR int in_counts_mRNA2_synthesis[] = {};
STATIC_VAR specie_id inputs_mRNA2_synthesis[] = {};
STATIC_VAR int out_counts_mRNA2_synthesis[] = {1};
STATIC_VAR specie_id outputs_mRNA2_synthesis[] = {mRNA2};
STATIC_VAR int num_factors_mRNA2_synthesis = 1;
STATIC_VAR specie_id factors_mRNA2_synthesis[] = {protein};


