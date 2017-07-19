#include "color.hpp"


namespace color
{
    namespace
    {
        bool iEnableColor = true;
    }
    
    void enable(const bool& pcfEnable)
    {
        iEnableColor = pcfEnable;
    }
    
    const std::string set(const unsigned int& pcfSetColor)
    {
        if (iEnableColor)
        {
            return "\x1b[3" + std::to_string(pcfSetColor) + "m";
        }
        else
        {
            return "";
        }
    }
    
    const std::string clear()
    {
        if (iEnableColor)
        {
            return "\x1b[0m";
        }
        else
        {
            return "";
        }
    }
};
