#include "csvr_param.hpp"

#include <iostream>
#include <string>


csvr_param::csvr_param(std::string const& pcfFileName) :
    csvr(pcfFileName) {}

param_set csvr_param::get_next() {
  param_set result;
  get_next(result);
  return result;
}


bool csvr_param::get_next(param_set& pfLoadTo) {
    if (!pfLoadTo.getArray()) {
      return false;
    }
    for (auto & parameter : pfLoadTo) {
      if (!csvr::get_next(&parameter)) break;
    }
    return true;
}
