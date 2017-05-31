#include <cstdio>
#include <fstream>
#include <string>

using namespace std;



// Gets the length of an array of type T
// Based on https://stackoverflow.com/a/3368894
template<typename T, const unsigned int size>
const unsigned int len(const T(&)[size])
{
    return size;
}



int main(int argc, char *argv[])
{
    if (argc == 2)
    {
        const string reaction[] = {
            #define REACTION(name) #name, 
            #define DELAY_REACTION(name) #name, 
            #include "reactions_list.hpp"
            #undef REACTION
            #undef DELAY_REACTION
        };
        
        const string delay[] = {
            #define REACTION(name) 
            #define DELAY_REACTION(name) #name, 
            #include "reactions_list.hpp"
            #undef REACTION
            #undef DELAY_REACTION
        };
        
        const string critical[] = {
            #define SPECIE(name)
            #define CRITICAL_SPECIE(name) #name, 
            #include "specie_list.hpp"
            #undef SPECIE
            #undef CRITICAL_SPECIE
        };
        
        
        
        // Create the new file along with comments documenting its format requirements
        string csv_name = string(argv[1])+"param_list_template.csv";
        ofstream csv_ofs(csv_name);
        if (csv_ofs.is_open())
        {
            // Write documentation
            csv_ofs << "# IMPORTANT: Rename this file to \"param_list.csv\" once data has been entered!\n#\n";
            csv_ofs << "# CSV Specification\n";
            
            csv_ofs << "#   Ignored by the simulation file reader are:\n";
            csv_ofs << "#     (1) blank rows and all other whitespace\n";
            csv_ofs << "#     (2) rows beginning with \'#\' -- these act as comments\n";
            csv_ofs << "#     (3) blank cells such as \"A, B, , D, E\"\n";
            csv_ofs << "#     (4) the first non-comment, non-blank row in the file*\n";
            
            csv_ofs << "#   *The cells of the first non-comment, non-blank row are column headers. These are provided for the user's convenience.\n";
            
            csv_ofs << "#   It is futile to modify the order of the column headers. Data must be entered in the order specified by the generated column headers in order for the simulation to work properly.\n\n";
            
            
            // Write column headers
            for (unsigned int i=0; i<len(reaction); i++)
            {
                csv_ofs << reaction[i] << ", ";
            }
            
            csv_ofs << ", ";
            
            for (unsigned int i=0; i<len(delay); i++)
            {
                csv_ofs << "dreact_" << delay[i] << ", ";
            }
            
            csv_ofs << ", ";
            
            for (unsigned int i=0; i<len(critical); i++)
            {
                csv_ofs << "rcrit_" << critical[i] << ", ";
            }
            
            
            // Victory message
            printf("\x1b[32mCSV column header generation successful. See \'%s\'\n", csv_name.c_str()); // green text
        }
        else // if output file stream failed to open
        {
            printf("\x1b[31mCSV column header generation failed. Could not create file \'%s\'\n", csv_name.c_str()); // red text
        }
        csv_ofs.close();
    }
    else // if argc != 2
    {
        printf("\x1b[31mCSV column header generation failed. Missing required command line argument:\n\t(1) relative directory containing desired model files, such as \"../models/her_model_2014/\", not including quotation marks\n"); // red text
    }
    
    printf("\x1b[0m"); // clear color
}
