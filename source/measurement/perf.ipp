template <typename Simulation>
PerfAnalysis<Simulation>::PerfAnalysis(std::vector<Species> const& observed_species,
  std::pair<dense::Natural, dense::Natural> cell_range,
  std::pair<Real, Real> time_range):
  Analysis<Simulation>(observed_species, cell_range, time_range)
  {
  finalized = false;
  ob=observed_species; 
  cr=cell_range;
  tr=time_range;
  }
  
template <typename Simulation>
void PerfAnalysis<Simulation>::update (Simulation & simulation, std::ostream& log) {
  perf_vector.emplace_back(simulation.get_step());
}

template <typename Simulation>
PerfAnalysis<Simulation>::PerfAnalysis (PerfAnalysis const& obj): Analysis<Simulation>(obj.ob, obj.cr, obj.tr){
  t = new runtimecheck;
}

template <typename Simulation>
void PerfAnalysis<Simulation>::finalize () {
    if (!finalized){
      finalized=true;
    }
    t->set_end();
    duration= t->get_duration(0,0);
    detail.concs.emplace_back(perf_vector.back()/duration);
}



template <typename Simulation>
void PerfAnalysis<Simulation>::show (csvw * csv_out) {
  Analysis<>::show(csv_out);
  if(csv_out){

      auto & out = *csv_out;
      out << "Performance :\n";
      for(int i : perf_vector){
        out<<perf_vector[i]<<"\n";
      }

  }

}


template<typename Simulation>
Details PerfAnalysis<Simulation>::get_details(){

        return detail;

}
