
template<int N, class T>
CUDA_AGNOSTIC
dense::cell_param<N, T>::cell_param(Natural width_total, Natural cells_total):
  cell_count_{cells_total},
  simulation_width_{width_total},
  _array{new T[_height * cell_count_]} {}
