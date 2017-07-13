/* build_once.cu
 * This file is a dummy file to include header files with static variable 
 * or inlineable function declarations that can only be included in a 
 * single .cpp or .cu file, but must be included in a .cpp file if 
 * compiling for CPU only, or in a .cu file if compiling for CPU/GPU 
 * heterogeneous execution.
 * It should be a functionally identical copy of build_once.cpp
 */

//Needed because static variable declarations need to be compiled differently
#include "reaction_deltas.hpp"

//Needed because without it, compilation of active rate functions on which 
//  determ_context methods depend won't be triggered
#include "model_impl.hpp"

//Needed because the context methods need to be cross-compiled for either CPU or GPU 
// execution, and their function signatures won't be the same in the two variants, which 
// would cause problems.
#include "sim/determ/determ_context.hpp"

//Needed because the reaction initializers need to be compiled differently depending 
//  on how the static variables they depend on (the deltas) are compiled.
#include "reaction_inits.hpp"

