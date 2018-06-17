#ifndef IO_CSVW_PARAM_HPP
#define IO_CSVW_PARAM_HPP

#include "csvw.hpp"


enum class param_type {
  SETS, PERT, GRAD
};

struct csvw_param : private csvw {
  csvw_param(std::string const& pcfFileName, param_type const& pcfType);
};

#endif
