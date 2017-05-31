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
        WHITE,
        RESET
    };
    
    extern bool enable;
    
    const std::string set(const unsigned int& setColor);
};

#endif // COLOR_HPP
