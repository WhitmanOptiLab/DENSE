// Struct for command line argument color

#ifndef COLOR_HPP
#define COLOR_HPP

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
