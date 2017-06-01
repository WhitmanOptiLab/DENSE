#include "color.hpp"


namespace color
{
    namespace
    {
        bool enableColor = true;
    }
    
    void enable(const bool& pcEnable)
    {
        enableColor = pcEnable;
    }
    
    const std::string set(const unsigned int& pcSetColor)
    {
        if (enableColor)
        {
            return "\x1b[3" + std::to_string(pcSetColor) + "m";
        }
        else
        {
            return "";
        }
    }
    
    const std::string clear()
    {
        return "\x1b[0m";
    }
};
