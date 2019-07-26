
template<int N, class T>
CUDA_AGNOSTIC
dense::cell_param<N, T>::cell_param(Natural width_total, Natural cells_total, Natural _num_growth_cells):
  cell_count_{cells_total},
  simulation_width_{width_total},
  _array{size_t(cell_count_ + _num_growth_cells)} {}
