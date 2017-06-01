#include "color.hpp"
#include "param_set.hpp"

#include <exception>
#include <iostream>
using namespace std;



// See documentation in header
ifstream param_set::current_ifstream;

// See documentation in header
unsigned int param_set::current_total = 0;
unsigned int param_set::current_remaining = 0;



// See usage documentation in header
bool param_set::open_ifstream(const string& pFileName)
{
    // If current_ifstream is already set to something, close it
    if (current_ifstream.is_open())
        close_ifstream();
    
    // Open file at pFileName
    current_ifstream.open(pFileName);
    
    // If open successful, count data sets and return true
    if (current_ifstream.is_open())
    {
        // Create a copy of the ifstream to be iterated through. This is because for some reason I had trouble resetting the stream back to the beginning using seekg(0) after the stream made its way to the end. By creating a copy, I don't have to mess with curren_ifstream's stream position.
        ifstream copy_current(pFileName);
        
        // Data lines counter
        current_total = current_remaining = 0;
        
        // The first char of each line should give away whether it's a data line or not. Count these data lines.
        char c = copy_current.get();
        while(copy_current.good())
        {
            if ((c >= '0' && c <= '9') || c == '.') // Only increment if it's a number or decimal
            {
                current_total++;
                current_remaining++;
            }
            
            // Skip entire or rest of line
            copy_current.ignore(unsigned(-1), '\n');
            c = copy_current.get();
            
            // "Trim" all excess whitespace
            while (c == ' ' || c == '\t' || c == '\n')
            {
                copy_current.ignore(unsigned(-1), c);
                c = copy_current.get();
            }
        }
        
        return true;
    }
    else
    {
        cout << color::set(color::RED) << "Failed to open \'" << pFileName << "\'." << color::clear() << endl;
        return false;
    }
}



// See usage documentation in header
void param_set::close_ifstream()
{
    current_ifstream.close();
    current_total = current_remaining = 0;
}



// See usage documentation in header
unsigned int param_set::get_set_total()
{
    return current_total;
}



// See usage documentation in header
unsigned int param_set::get_set_remaining()
{
    return current_remaining;
}



// See usage documentation in header
param_set param_set::load_next_set()
{
    param_set set_return;
    load_next_set(set_return);
    return set_return;
}



bool param_set::load_next_set(param_set &pLoadTo)
{
    bool load_success = false;
    
    // Double check if current_ifstream is actually open
    if (current_ifstream.is_open())
    {
        // Param data from file to be pushed
        string param;
        unsigned int param_index = 0;
        unsigned int array_num = 0;
        
        // For error reporting
        unsigned int line_num = 1;
        
        // Keep track of where we are in the file
        // If we already parsed a set, we've already gotten past the header and tellg() would return something greater than 0, i.e. true
        bool past_header = int(current_ifstream.tellg());
        
        char c = current_ifstream.get();
        while(current_ifstream.good())
        {
            if (c == '#') // Check if in comment
            {
                // Skip comment line
                current_ifstream.ignore(unsigned(-1), '\n');
                line_num++;
            }
            else if (c != ' ' && c != '\t') // Parse only if not whitespace except for \n
            {
                if (!past_header)
                {
                    // Wait for column header line before kick-starting the parser
                    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_')
                    {
                        past_header = true;
                        // Skip column header line
                        current_ifstream.ignore(unsigned(-1), '\n');
                        line_num++;
                    }
                }
                else
                {
                    // If hit data seperator, add data to respective array
                    // '\n' is there in case there is no comma after last item
                    if (c == ',' || c == '\n')
                    {
                        if (param.length() > 0) // Only push if param contains something
                        {
                            RATETYPE *array_ptr = nullptr;
                            
                            switch (array_num)
                            {
                                // This ordering must match that of gen_csv.cpp
                            case 0:
                                array_ptr = pLoadTo._rates_base;
                                if (param_index > NUM_REACTIONS - 1)
                                {
                                    param_index = 0;
                                    array_num++;
                                }
                                else
                                {
                                    break;
                                }
                            case 1:
                                array_ptr = pLoadTo._delay_sets;
                                if (param_index > NUM_DELAY_REACTIONS - 1)
                                {
                                    param_index = 0;
                                    array_num++;
                                }
                                else
                                {
                                    break;
                                }
                            case 2:
                                array_ptr = pLoadTo._critical_values;
                                if (param_index > NUM_CRITICAL_SPECIES - 1)
                                {
                                    param_index = 0;
                                    array_num++;
                                }
                                else
                                {
                                    break;
                                }
                            default:
                                array_ptr = nullptr;
                            }
                            
                            
                            // If array_ptr is set to something, try adding data to array_ptr
                            if (array_ptr != nullptr)
                            {
                                try
                                {
                                    array_ptr[param_index++] = stold(param);
                                    !load_success ? load_success = true : 0;
                                }
                                catch(exception ex) // For catching stold() errors
                                {
                                    cout << color::set(color::RED) << "CSV parsing failed. Invalid data contained at line " << line_num << "." << color::clear() << endl;
                                    load_success = false;
                                    break;
                                }
                            }
                            
                            param.clear();
                        }
                        else if (c == '\n') // if param was empty and it's a new line, we're done parsing the set
                        {
                            break;
                        }
                    }
                    else if ((c >= '0' && c <= '9') || c == '.') // Parse if it is numbers or decimal
                    {
                        param += c;
                    }
                }
            }
            
            // increment line counter
            if (c == '\n')
            {
                line_num++;
            }
            
            // get next char in file
            c = current_ifstream.get();
        }
    }
    else // if failed to open current_ifstream
    {
        cout << color::set(color::RED) << "CSV parsing failed. No CSV file has been chosen." << color::clear() << endl;
        load_success = false;
    }
    
    // Decrement current_remaining if successful
    load_success ? current_remaining-- : 0;
    
    return load_success;
}

