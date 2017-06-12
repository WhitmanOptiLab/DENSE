#ifndef ARG_PARSE_HPP
#define ARG_PARSE_HPP

/*
 * Why a namespace?! Fascinating read on the dangers of the "private static
 *   member solution": https://stackoverflow.com/a/112451
 *
 * This class was going to be entirely full of static functions and fields
 *   anyway, so we might as well make it a namespace!
 *
 * What do the variable prefixes mean???
 *   u - public / visible (namespace)
 *   o - protected
 *   i - private / anonymous (namespace)
 *   s - static
 *   p - parameter
 *   r - return
 *   t - temporary holder variable
 *   h - push (to a container/array)
 *   c - constant
 *   f - reference
 *   n - pointer
 *   l - iterator (looper)
 *   g - general/other -- only use when NO other prefix applies; never use with another prefix
 * Append prefixes in the order they are listed in.
*/

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
    void init(const int& pcfArgc, char* pcfArgv[]);
    
    
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
    const T get(const std::string& pcfFlagShort, const std::string& pcfFlagLong);
    template<typename T>
    const T get(const std::string& pcfFlagShort, const std::string& pcfFlagLong, const T& pcfDefault);
};

#endif // ARG_PARSE_HPP
