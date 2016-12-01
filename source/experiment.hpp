#ifndef EXPERIMENT_HPP
#define EXPERIMENT_HPP



using namespace std;

class experiment{
private:
    char* params_file; // The path and name of the parameter sets file, default=input.params
    bool read_params; // Whether or not the read the parameter sets file, default=false
    char* ranges_file; // The path and name of the parameter ranges file, default=none
    bool read_ranges; // Whether or not to read the ranges file, default=false
    char* perturb_file; // The path and name of the perturbations file, default=none
    bool read_perturb; // Whether or not to read the perturbations file, default=false
    char* gradients_file; // The path and name of the gradients file, default=none
    bool read_gradients; // Whether or not to read the gradients file, default=false
    char* passed_file; // The path and name of the passed file, default=output.passed
    bool print_passed; // Whether or not to print the passed file, default=false
    char* dir_path; // The path of the output directory for concentrations or oscillation features, default=none
    bool print_cons; // Whether or not to print concentrations, default=false
    bool binary_cons_output; // Whether or not to print the binary or ASCII value of numbers in the concentrations output files
    char* features_file; // The path and file of the features file, default=none
    bool ant_features; // Whether or not to print oscillation features in the anterior
    bool post_features; // Whether or not to print oscillation features in the posterior
    bool print_features; // Whether or not to print general features file, default=false
    char* conditions_file; // The path and file of the conditions file, default=none
    bool print_conditions; // Whether or not to print which conditions passed, default=false
    char* scores_file; // The path and file of the scores file, default=none
    bool print_scores; // Whether or not to print the scores for every mutant, default=false
    int num_colls_print; // The number of columns of cells to print for plotting of single cells on top of each other
};

#endif

