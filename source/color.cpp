#include "color.hpp"


namespace color
{
    bool enable = true;
    
    const std::string set(const unsigned int& setColor)
    {
        if (!enable || (setColor == RESET))
        {
            return "\x1b[0m";
        }
        else
        {
            return "\x1b[3" + std::to_string(setColor) + "m";
        }
    }
};
