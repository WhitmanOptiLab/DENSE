// Struct for command line argument color

#ifndef UTIL_COLOR_HPP
#define UTIL_COLOR_HPP

#include <string>


namespace color
{
    enum
    {
        BLACK,
        RED,
        GREEN,
        YELLOW,
        BLUE,
        MAGENTA,
        CYAN,
        WHITE
    };
    
    void enable(const bool& pcfEnable);
    const std::string set(const unsigned int& pcfSetColor);
    const std::string clear();
};

#endif // COLOR_HPP
