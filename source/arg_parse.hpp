/*
 * Why a namespace?! Fascinating read on the dangers of the "private static
 *   member solution": https://stackoverflow.com/a/112451
 *
 * This class was going to be entirely full of static functions and fields
 *   anyway, so we might as well make it a namespace!
*/

#ifndef ARG_PARSE_HPP
#define ARG_PARSE_HPP

#include <string>

namespace arg_parse
{
    /**
     *  Initialize arg_parse
     *  
     *  usage
     *      Must be called, ideally early in "int main(...)", before using "const T get(...)"
     *  
     *  parameters
     *      pcArgc - argc of "int main(int argc, char *argv[])"
     *      pcArgv - argv of "int main(int argc, char *argv[])"
    */
    void init(const int& pcArgc, char* pcArgv[]);
    
    
    /**
     *  Get Command Line flag Value or Return Default Value
     *  
     *  usage
     *      Anywhere in your program after init(...) is called, return the value proceeding "-pcFlagShort" or "--pcFlagLong"
     *      typename T must be either std::string, bool, int, or RATETYPE
     *  
     *  parameters
     *      pcFlagShort - short version of command line flag, do not include "-" at beginning
     *      pcFlagLong - long version of command line flag, do not include "--" at beginning
     *      pcDefault - default value to return if either pcFlagShort or pcFlagLong is not found
     *  
     *  returns
     *      Value from command line in type T proceeding flag
     *      If typename is not of std::string, bool, int, or RATETYPE, prints an error to the command line and returns nullptr
     *  
     *  notes
     *      BEFORE CALLING THIS FUNCTION, PLEASE CALL "arg_parse::init(...)"
     *      Flag Standards
     *          bool flags have upper-case pcFlagShort while all else have lowercase
     *          Seperate words of pcFlagLong by single dashes "-" as in "file-name"
     *      Version without pcDefault means that the flag is obligatory (unless typename is bool) and will print warning message to user if not present in argv
    */
    template<typename T>
    const T get(const std::string& pcFlagShort, const std::string& pcFlagLong);
    template<typename T>
    const T get(const std::string& pcFlagShort, const std::string& pcFlagLong, const T& pcDefault);
};

#endif // ARG_PARSE_HPP
