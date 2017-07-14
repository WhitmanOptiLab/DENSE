#include "csvw_param.hpp"
#include "util/color.hpp"
#include "core/specie.hpp"

#include <iostream>
#include <string>
using namespace std;



csvw_param::csvw_param(const string& pcfFileName, const param_type& pcfType) :
    csvw::csvw(pcfFileName, true, "# Rename this file by removing the "
            "\"_template\" from the file name (or just change the name "
            "entirely) once the data has been entered!\n")
{
    const string cReaction[] = {
        #define REACTION(name) #name, 
        #define DELAY_REACTION(name) #name, 
        #include "reactions_list.hpp"
        #undef REACTION
        #undef DELAY_REACTION
    };
    
    const string cDelay[] = {
        #define REACTION(name) 
        #define DELAY_REACTION(name) #name, 
        #include "reactions_list.hpp"
        #undef REACTION
        #undef DELAY_REACTION
    };
    
    const string cCritical[] = {
        #define SPECIE(name)
        #define CRITICAL_SPECIE(name) #name, 
        #include "specie_list.hpp"
        #undef SPECIE
        #undef CRITICAL_SPECIE
    };


    // param_type string prefix
    string **nPrefix;
    string param_type_str;
    switch (pcfType)
    {
        case param_type::SETS:
            csvw::add_div("# This file can contain more than one "
                    "set (each being on their own line). All sets "
                    "are initialized and executed in parallel when a file is "
                    "loaded into the simulation.\n");
            csvw::add_div("# For more information and examples see README.md "
                    "section 2.2.1\n\n");
            param_type_str = "sets";
            nPrefix = new string*[2];
            nPrefix[0] = new string("");
            nPrefix[1] = 0;
            break;
        case param_type::PERT:
            csvw::add_div("# This file should only contain one set of "
                    "perturbations. Only this one perturbations set is applied "
                    "to all parameter sets when a simulation set is being run.\n");
            csvw::add_div("# Use \'0\' to indicate that a "
                    "reaction should not have perturbations.\n");
            csvw::add_div("# For more information and examples see README.md "
                    "section 2.2.2\n\n");
            param_type_str = "perturbations";
            nPrefix = new string*[2];
            nPrefix[0] = new string("pert_");
            nPrefix[1] = 0;
            break;
        case param_type::GRAD:
            csvw::add_div("# This file should only contain one set of "
                    "gradients. Only this one gradients setting is applied "
                    "to all parameter sets when a simulation set is being run.\n");
            csvw::add_div("# Use \'0\' under all four columns of a reaction "
                    "to indiate that it should not have a gradient.\n"
                    "# Gradient Codes\n"
                    "#   x1 - start column\n"
                    "#   y1 - start multiplier (use \'1.23\' to mean \'123%\')\n"
                    "#   x2 - end column\n"
                    "#   y2 - end multiplier\n");
            csvw::add_div("# For more information and examples see README.md "
                    "section 2.2.2\n\n");
            param_type_str = "gradients";
            nPrefix = new string*[5];
            nPrefix[0] = new string("grad_x1_");
            nPrefix[1] = new string("grad_y1_");
            nPrefix[2] = new string("grad_x2_");
            nPrefix[3] = new string("grad_y2_");
            nPrefix[4] = 0;
            break;
    }
    string **lnPrefix = nPrefix;

    
    
    // Write column headers
    string zeros_line = "\n";
    for (unsigned int i=0; i<NUM_REACTIONS; i++)
    {
        do
        {
            csvw::add_div(**lnPrefix + cReaction[i] + ", ");
            zeros_line += "0, ";
        } while (*(++lnPrefix)); // Iterate over prefixes
        lnPrefix = nPrefix; // Reset prefix pointer to beginning
        
        // Add extra comma between gradient params for readability
        if (pcfType == param_type::GRAD)
        {
            csvw::add_div(", ");
            zeros_line += ", ";
        }
    }
    
    csvw::add_div(", ");
    zeros_line += ", ";

    for (unsigned int i=0; i<NUM_DELAY_REACTIONS; i++)
    {
        do
        {
            csvw::add_div(**lnPrefix + "dreact_" + cDelay[i] + ", ");
            zeros_line += "0, ";
        } while (*(++lnPrefix));
        lnPrefix = nPrefix;
        
        if (pcfType == param_type::GRAD)
        {
            csvw::add_div(", ");
            zeros_line += ", ";
        }
    }
    
    csvw::add_div(", ");
    zeros_line += ", ";

    for (unsigned int i=0; i<NUM_CRITICAL_SPECIES; i++)
    {
        do
        {
            csvw::add_div(**lnPrefix + "rcrit_" + cCritical[i] + ", ");
            zeros_line += "0, ";
        } while (*(++lnPrefix));
        lnPrefix = nPrefix;

        if (pcfType == param_type::GRAD)
        {
            csvw::add_div(", ");
            zeros_line += ", ";
        }
    }
   

    if (pcfType == param_type::PERT || pcfType == param_type::GRAD)
    {
        csvw::add_div(zeros_line);
    }

    
    // Victory message
    cout << color::set(color::GREEN) << "CSV parameter " << param_type_str <<
        " column header generation successful. See \'" << pcfFileName << "\'."
        << color::clear() << endl;
}


csvw_param::~csvw_param()
{
}
