#include "arg_parse.hpp"
#include "utility/style.hpp"
#include "utility/common_utils.hpp"
#include "core/specie.hpp"
#include "utility/numerics.hpp"

using style::Color;

#include <iostream>
#include <vector>




namespace arg_parse
{
    // anonymous namespace
    namespace
    {
        // For storing a copy of *argv[]
        std::vector<std::string> iArgVec;

        // Stops obligatory message
        bool iSuppressObligatory = false;


        // Get index of (or index after) pcFlag if it exists in iArgVec.
        // Return false if not found.
        bool getIndex(std::string pcfFlagShort, std::string pcfFlagLong,
                int* pnIndex, bool const& pcfNext)
        {
            pcfFlagShort = "-" + pcfFlagShort;
            pcfFlagLong = "--" + pcfFlagLong;

            for (std::size_t i = 0; i < iArgVec.size(); i++)
            {
                if (iArgVec[i] == pcfFlagShort || iArgVec[i] == pcfFlagLong)
                {
                    if (pcfNext)
                    {
                        if (i + 1 >= iArgVec.size())
                        {
                            std::cout << style::apply(Color::red) << "Command line "
                                "argument search failed. No argument provided "
                                "after flag [" << pcfFlagShort << " | " <<
                                pcfFlagLong << "]." << style::reset() << '\n';
                        }
                        else
                        {
                            if (pnIndex)
                                *pnIndex = i + 1;
                            return true;
                        }
                    }
                    else
                    {
                        if (pnIndex)
                            *pnIndex = i;
                        return true;
                    }
                }
            }

            return false;
        }


        // Prints message warning that flag is required
        void warnObligatory(std::string pcfFlagShort, std::string pcfFlagLong)
        {
            if (!iSuppressObligatory)
            {
                std::cout << style::apply(Color::red) << "Command line argument "
                    "search failed. Flag [-" << pcfFlagShort << " | --" <<
                    pcfFlagLong << "] is required in order for all program "
                    "behaviors to function properly." <<
                    style::reset() << '\n';
            }
        }


        template<typename T>
        const T getNotOblig(std::string const& pcfFlagShort,
                std::string const& pcfFlagLong)
        {
            iSuppressObligatory = true;
            const T rval = get<T>(pcfFlagShort, pcfFlagLong);
            iSuppressObligatory = false;
            return rval;
        }

    } // end anonymous namespace




    // See usage documentation in header
    void init(int const& argc, char* argv[]) {
      // file_name = argv[0]
      iArgVec.assign(argv + 1, argv + argc);
    }




    template<>
    bool get<std::string>(std::string const& pcfFlagShort,
            std::string const& pcfFlagLong, std::string *  pnPushTo,
            bool const& pcfObligatory)
    {
        int index;
        if (getIndex(pcfFlagShort, pcfFlagLong, &index, true))
        {
            if (pnPushTo)
                *pnPushTo = iArgVec[index];
            return true;
        }
        else
        {
            if (pcfObligatory)
                warnObligatory(pcfFlagShort, pcfFlagLong);
            return false;
        }
    }

    // The default is a vec filled with all specie ids
    template<>
    bool get<std::vector<Species>>(std::string const& pcfFlagShort,
            std::string const& pcfFlagLong, std::vector<Species>* pnPushTo,
            bool const& pcfObligatory)
    {
        auto rVec = str_to_species(
                get<std::string>(pcfFlagShort, pcfFlagLong, ""));

        if (!rVec.empty())
        {
            if (pnPushTo)
                *pnPushTo = rVec;
            return true;
        }
        else
        {
            if (pcfObligatory)
                warnObligatory(pcfFlagShort, pcfFlagLong);
            return false;
        }
    }

    template<>
    bool get<int>(std::string const& pcfFlagShort, std::string const& pcfFlagLong,
            int* pnPushTo, bool const& pcfObligatory)
    {
        bool success = true;
        int index;
        if (getIndex(pcfFlagShort, pcfFlagLong, &index, true))
        {
            char* tInvalidAt;
            int tPushTo = strtol(iArgVec[index].c_str(), &tInvalidAt, 0);

            if (*tInvalidAt)
            {
                success = false;
                std::cout << style::apply(Color::red) << "Command line argument "
                    "parsing failed. Argument \"" << iArgVec[index] <<
                    "\" cannot be converted to integer." <<
                    style::reset() << '\n';
            }
            else
            {
                *pnPushTo = tPushTo;
            }
        }
        else
        {
            if (pcfObligatory)
                warnObligatory(pcfFlagShort, pcfFlagLong);
            success = false;
        }

        return success;
    }

    template<>
    bool get<Real>(std::string const& pcfFlagShort, std::string const& pcfFlagLong,
            Real* pnPushTo, bool const& pcfObligatory)
    {
        bool success = true;
        int index;
        if (getIndex(pcfFlagShort, pcfFlagLong, &index, true))
        {
            char* tInvalidAt;
            Real tPushTo = strtold(iArgVec[index].c_str(), &tInvalidAt);

            if (*tInvalidAt)
            {
                success = false;
                std::cout << style::apply(Color::red) << "Command line argument "
                    "parsing failed. Argument \"" << iArgVec[index] <<
                    "\" cannot be converted to Real." <<
                    style::reset() << '\n';
            }
            else
            {
                *pnPushTo = tPushTo;
            }
        }
        else
        {
            if (pcfObligatory)
                warnObligatory(pcfFlagShort, pcfFlagLong);
            success = false;
        }

        return success;
    }

    template<>
    bool get<bool>(std::string const& pcfFlagShort, std::string const& pcfFlagLong,
            bool* pnPushTo, bool const& pcfObligatory)
    {
        // true if found, false if not
        bool found = getIndex(pcfFlagShort, pcfFlagLong, nullptr, false);

        if (pnPushTo)
            *pnPushTo = found;
        return found;
    }



    /*
    template<>
    bool get<bool>(std::string const& pcfFlagShort, std::string const& pcfFlagLong,
            bool const& pcfDefault)
    {
        if (getIndex(pcfFlagShort, pcfFlagLong, 0, false)) // If found
            return !pcfDefault;
        else
            return pcfDefault;
    }
    */
}
