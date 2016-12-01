#ifndef REACTION_HPP
#define REACTION_HPP



using namespace std;

class reaction{
    
private:

    
    //di_indices
    int con_protein_self; // The index of this protein's concentrations
    int con_protein_other; // The index of the interacting protein's concentrations
    int con_dimer; // The index of this protein's and the interacting one's heterodimer's concentrations
    int rate_association; // The index of the heterodimer's rate of association
    int rate_dissociation; // The index of the heterodimer's rate of dissociation
    int dimer_effect; // The index in the dimer_effects array in the associated di_args struct
    
    
    
    /* cph_indices contains indices for con_protein_her, the Her protein concentration function
     notes:
     This struct is just a wrapper used to minimize the number of arguments passed into con_protein_her.
     todo:
     */
    //cph_indices
    int con_mrna; // The index of this protein's associated mRNA concentrations
    int con_protein; // The index of this protein's concentrations
    int con_dimer; // The index of this protein's homodimer's concentrations
    int rate_synthesis; // The index of this protein's rate of synthesis
    int rate_degradation; // The index of this protein's rate of degradation
    int rate_association; // The index of the homodimer's rate of association
    int rate_dissociation; // The index of the homodimer's rate of dissociation
    int delay_protein; // The index of this protein's rate of delay
    int dimer_effect; // The index in the dimer_effects array in the associated cp_args struct
    int old_cell; // The index in the old_cells array in the associated cp_args struct
    
    
    /* cpd_indices contains indices for con_protein_delta, the Delta protein concentration function
     notes:
     This struct is just a wrapper used to minimize the number of arguments passed intocon_protein_delta.
     todo:
     */
    //cpd_indices
    int con_mrna; // The index of this protein's associated mRNA concentrations
    int con_protein; // The index of this protein's concentrations
    int rate_synthesis; // The index of this protein's rate of synthesis
    int rate_degradation; // The index of this protein's rate of degradation
    int delay_protein; // The index of this protein's rate of delay
    int old_cell; // The index in the old_cells array in the associated cp_args struct
    
    
    
    
    /* cd_indices contains indices for con_dimer, the dimer concentration function
     notes:
     This struct is just a wrapper used to minimize the number of arguments passed into con_dimer.
     todo:
     */
    //cd_indices
    int con_protein; // The index of this dimer's constituent protein concentration
    int rate_association; // The index of this dimer's rate of association
    int rate_dissociation; // The index of this dimer's rate of dissociation
    int rate_degradation; // The index of this dimer's rate of degradation
    
    
public:
};


#endif

