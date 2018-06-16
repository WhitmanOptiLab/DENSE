#include "io/csvw_param.hpp"
#include "utility/style.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

int main (int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << style::apply(Color::red) <<
      "CSV parameter list column header generation failed. Missing required command line argument:\n\t(1) Relative directory containing desired model files, such as \"../models/her_model_2014/\", not including quotation marks. Spaces not allowed in file/directory names. For indicating the current directory, use \"./\".\n" <<
      style::reset();
    return EXIT_FAILURE;
  }

  std::string dir = argv[1];
  csvw_param( dir + "param_sets_template.csv", param_type::SETS );
  csvw_param( dir + "param_pert_template.csv", param_type::PERT );
  csvw_param( dir + "param_grad_template.csv", param_type::GRAD );
  return EXIT_SUCCESS;
}
