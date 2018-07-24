#include "utility/style.hpp"

using style::Color;

#include <cstdlib>
#include <iostream>
#include <string>

#include "io/csvw.hpp"

enum class param_type {
  SETS, PERT, GRAD
};

struct csvw_param : private csvw {
  csvw_param(std::string const& pcfFileName, param_type const& pcfType);
};

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


#include "utility/style.hpp"
#include "core/specie.hpp"
#include "core/reaction.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

    const std::string cReaction[] = {
        #define REACTION(name) #name,
        #define DELAY_REACTION(name) #name,
        #include "reactions_list.hpp"
        #undef REACTION
        #undef DELAY_REACTION
    };

    const std::string cDelay[] = {
        #define REACTION(name)
        #define DELAY_REACTION(name) "dreact_" #name,
        #include "reactions_list.hpp"
        #undef REACTION
        #undef DELAY_REACTION
    };

    const std::string cCritical[] = {
        #define SPECIE(name)
        #define CRITICAL_SPECIE(name) "rcrit_" #name,
        #include "specie_list.hpp"
        #undef SPECIE
        #undef CRITICAL_SPECIE
    };

}

csvw_param::csvw_param(std::string const& pcfFileName, param_type const& pcfType) :
    csvw::csvw(pcfFileName)
{
    csvw & out = *this;


    out <<
          "# CSV Specification\n"
          "#   Ignored by the file readers are:\n"
          "#     (1) Empty cells / Blank rows / Whitespace\n"
          "#     (2) Comment whose rows always begin with a \'#\'\n"
          "#         For best results delete all comments before loading "
            "this file into any Excel-like program.\n"
          "#     (3) Any cell which does not conform to the scientific "
            "notation format 3.14e-41 or simple whole numbers and decimals\n"
          "#         Often times cells which do not contain numbers are "
            "intended to be column headers. These are not parsed by the simulation "
            "and can technically be modified by the users as they wish.\n"
          "#         It is futile to add/remove/modify the column headers "
            "with the expectation of changing the program's behavior. Data must "
            "be entered in the default order for it to be parsed correctly.\n"
          "# None of these comments include commas because it messes with "
            "the column widths when loaded into Excel-like programs.\n"
          "# For more information and examples see README.md section 2.2.0\n";

    out << "\n# Rename this file by removing the "
            "\"_template\" from the file name (or just change the name "
            "entirely) once the data has been entered!\n";

    // param_type string prefix
    std::vector<std::string> nPrefix;
    std::string param_type_str;
    switch (pcfType)
    {
        case param_type::SETS:
            out << ("# This file can contain more than one "
                    "set (each being on their own line). All sets "
                    "are initialized and executed in parallel when a file is "
                    "loaded into the simulation.\n");
            out << ("# For more information and examples see README.md "
                    "section 2.2.1\n\n");
            param_type_str = "sets";
            nPrefix = { "" };
            break;
        case param_type::PERT:
            out << ("# This file should only contain one set of "
                    "perturbations. Only this one perturbations set is applied "
                    "to all parameter sets when a simulation set is being run.\n");
            out << ("# Use \'0\' to indicate that a "
                    "reaction should not have perturbations.\n");
            out << ("# For more information and examples see README.md "
                    "section 2.2.2\n\n");
            param_type_str = "perturbations";
            nPrefix = { "pert_" };
            break;
        case param_type::GRAD:
            out << ("# This file should only contain one set of "
                    "gradients. Only this one gradients setting is applied "
                    "to all parameter sets when a simulation set is being run.\n");
            out << ("# Use \'0\' under all four columns of a reaction "
                    "to indiate that it should not have a gradient.\n"
                    "# Gradient Codes\n"
                    "#   x1 - start column\n"
                    "#   y1 - start multiplier (use \'1.23\' to mean \'123%\')\n"
                    "#   x2 - end column\n"
                    "#   y2 - end multiplier\n");
            out << ("# For more information and examples see README.md "
                    "section 2.2.2\n\n");
            param_type_str = "gradients";
            nPrefix = { "grad_x1_", "grad_y1_", "grad_x2_", "grad_y2_" };
            break;
        default: throw std::out_of_range("Invalid param_type: " + static_cast<int>(pcfType));
    }

    // Write column headers
    std::string zeros_line = "\n";
    for (unsigned int i = 0; i < NUM_REACTIONS; ++i) {
        for (auto & prefix : nPrefix) {
            out << prefix << cReaction[i] << ", ";
            zeros_line += "0, ";
        }
        // Add extra comma between gradient params for readability
        if (pcfType == param_type::GRAD) {
            out << ", ";
            zeros_line += ", ";
        }
    }

    out << ", ";
    zeros_line += ", ";

    for (unsigned int i = 0; i < NUM_DELAY_REACTIONS; ++i)
    {
        for (auto & prefix : nPrefix) {
            out << prefix << cDelay[i] << ", ";
            zeros_line += "0, ";
        }

        if (pcfType == param_type::GRAD) {
            out << ", ";
            zeros_line += ", ";
        }
    }

    out << ", ";
    zeros_line += ", ";

    for (unsigned int i = 0; i < NUM_CRITICAL_SPECIES; i++) {
        for (auto & prefix : nPrefix) {
            out << prefix << "rcrit_" << cCritical[i] << ", ";
            zeros_line += "0, ";
        }

        if (pcfType == param_type::GRAD) {
            out << ", ";
            zeros_line += ", ";
        }
    }


    if (pcfType == param_type::PERT || pcfType == param_type::GRAD) {
      out << (zeros_line);
    }

    std::cout << style::apply(Color::green) <<
      "CSV parameter " << param_type_str <<
      " column header generation successful. " <<
      "See \'" << pcfFileName << "\'.\n" << style::reset();
}
