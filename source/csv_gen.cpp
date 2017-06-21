#include "csvw_param.hpp"
#include "color.hpp"

#include <iostream>
#include <string>
using namespace std;



int main(int argc, char *argv[])
{
    if (argc == 2)
    {
        string fileName = string(argv[1]) + "param_list_template.csv";
        ifstream openCheck(fileName);
        
        // Only generate if not already exist
        if (!openCheck.is_open())
        {
            csvw_param csvwp(fileName);
        }
        else
        {
            cout << color::set(color::YELLOW) << "CSV parameter list column header generator not executed; a copy of param_list_template.csv already exists in \"" << argv[1] << "\"." << color::clear() << endl;
            openCheck.close();
        }
    }
    else // if argc != 2
    {
        cout << color::set(color::RED) << "CSV parameter list column header generation failed. Missing required command line argument:\n\t(1) Relative directory containing desired model files, such as \"../models/her_model_2014/\", not including quotation marks. Spaces not allowed in file/directory names. For indicating the current directory, use \"./\".\n" << color::clear() << endl;
    }
}
