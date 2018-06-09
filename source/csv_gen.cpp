#include "io/csvw_param.hpp"
#include "util/color.hpp"

#include <iostream>
#include <string>
#include <utility>

int main(int argc, char *argv[])
{
    if (argc == 2)
    {
        std::string dir {argv[1]};
        csvw_param( dir + "param_sets_template.csv", param_type::SETS );
        csvw_param( dir + "param_pert_template.csv", param_type::PERT );
        csvw_param( dir + "param_grad_template.csv", param_type::GRAD );
    }
    else // if argc != 2
    {
        std::cout << color::set(color::RED) << "CSV parameter list column header generation failed. Missing required command line argument:\n\t(1) Relative directory containing desired model files, such as \"../models/her_model_2014/\", not including quotation marks. Spaces not allowed in file/directory names. For indicating the current directory, use \"./\".\n" << color::clear();
    }
}
