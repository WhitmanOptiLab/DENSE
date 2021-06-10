#ifndef ARG_HPP
#define ARG_HPP

#include "io/arg_parse.hpp"
#include "measurement/oscillation.hpp"
#include "measurement/basic.hpp"
#include "measurement/bad_simulation_error.hpp"
#include "utility/style.hpp"
#include "utility/common_utils.hpp"
#include "io/csvr_sim.hpp"
#include "model_impl.hpp"
#include "io/ezxml/ezxml.h"
#include "ngraph/ngraph_components.hpp"
#include "Sim_Initializer.hpp"

using style::Color;

#include <chrono>
#include <memory>
#include <iterator>
#include <algorithm>
#include <functional>
#include <exception>
#include <iostream>
#include <cmath>

using dense::Static_Args;
using dense::Param_Static_Args;

namespace dense {

void display_usage(std::ostream& out);
Static_Args parse_static_args(int argc, char* argv[]);
void display_param_search_usage(std::ostream& out);
Param_Static_Args param_search_parse_static_args(int argc, char* argv[]);

}

#endif
