/*
 * Why a namespace?! Fascinating read on the dangers of the "private static
 *   member solution": https://stackoverflow.com/a/112451
 *
 * This class was going to be entirely full of static functions and fields
 *   anyway, so we might as well make it a namespace!
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
    void init(const int& pcArgc, char* pcArgv[]);
    
    
    /**
     *  Get Command Line Option Value or Return Default Value
     *  
     *  usage
     *      Anywhere in your program after init(...) is called, return the value proceeding "-pcOptShort" or "--pcOptLong"
     *      typename T must be either std::string, bool, int, or RATETYPE
     *  
     *  parameters
     *      pcOptShort - short version of command line tag, do not include "-" at beginning
     *      pcDefault - default value to return if either pcOptShort or pcOptLong is not found
     *      pcOptLong - long version of command line tag, do not include "--" at beginning
     *  
     *  returns
     *      Value of typename T proceeding indicated command line tags/options
     *      If typename is not of std::string, bool, int, or RATETYPE, prints an error to the command line and returns nullptr
     *  
     *  notes
     *      BEFORE CALLING THIS FUNCTION, PLEASE CALL "arg_parse::init(...)"
    */
    template<typename T>
    const T get(const std::string& pcOptShort, const T& pcDefault, const std::string& pcOptLong = "");
};
