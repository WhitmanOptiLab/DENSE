#ifndef IO_ARG_PARSE_HPP
#define IO_ARG_PARSE_HPP

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
 *   l - iterator (looper)
 *   h - push (to a container/array)
 *   c - constant
 *   f - reference
 *   n - pointer
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
    void init(int const& pcfArgc, char* pcfArgv[]);


    /**
     *  Get Command Line flag Value or Return Default Value
     *
     *  usage
     *      Anywhere in your program after init(...) is called, return the value proceeding "-pcFlagShort" or "--pcFlagLong"
     *      IMPORTANT!!! typename T must be either std::string, bool, int, Real, or std::vector<Species>
     *      Second version of get<>() does not work with std::vector<Species> typename
     *
     *  parameters
     *      pcFlagShort - short version of command line flag, do not include "-" at beginning
     *      pcFlagLong - long version of command line flag, do not include "--" at beginning
     *      pnPushTo - pointer to a variable whose value you want set to that of the argument
     *      pcfObligatory - if set to true and the flags aren't found, will output warning message saying that those flags are required
     *      pcDefault - default value to return if either pcFlagShort or pcFlagLong is not found
     *
     *  returns
     *      Value from command line in type T proceeding flag
     *      If typename is not of std::string, bool, int, or Real, prints an error to the command line and returns nullptr
     *
     *  notes
     *      BEFORE CALLING THIS FUNCTION, PLEASE CALL "arg_parse::init(...)"
     *      Flag Standards
     *          bool flags have upper-case pcFlagShort while all else have lowercase
     *          Seperate words of pcFlagLong by single dashes "-" as in "file-name"
     *      get(...) without pcfDefault DOES NOT EXIST FOR GOOD REASON. It is optimal to use the first version of get<>() because one can control the flow of the program when it encounters missing required arguments, preventing pesky crashes and infinite loops.
    */
    template<typename T>
    bool get(std::string const& pcfFlagShort, std::string const& pcfFlagLong, T* pnPushTo, bool const& pcfObligatory);
    template<typename T>
    T get(std::string const& pcfFlagShort, std::string const& pcfFlagLong, T const& pcfDefault)
    {
        T rVar = pcfDefault;
        // Not obligatory, explained in "notes"
        get<T>(pcfFlagShort, pcfFlagLong, &rVar, false);
        return rVar;
    }

}

#endif // ARG_PARSE_HPP
