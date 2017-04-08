#include "model.hpp"

model::model(bool using_gradients, bool using_perturb) :
  _using_perturb(using_perturb),
  _using_gradients(using_gradients) {
  for (int i = 0; i < NUM_REACTIONS; i++) {
    factors_perturb[i] = 0.0;
    _has_gradient[i] = false;
    factors_gradient[i] = NULL;
  }
}
