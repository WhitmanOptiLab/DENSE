#ifndef DENSE_BAD_SIMULATION_ERROR
#define DENSE_BAD_SIMULATION_ERROR

#include <stdexcept>

namespace dense {

  /// An exception thrown to reject a simulation for having produced data that
  /// is not realistic enough to be meaningfully analyzed.
  #ifndef __cpp_concepts
  template <typename Simulation>
  #else
  template <Simulation_Concept Simulation>
  #endif
  class Bad_Simulation_Error : public std::runtime_error {

    public:

      /// Construct a Bad_Simulation_Error from an explanatory string and
      /// a reference to the simulation being rejected.
      ///
      /// \param explanation  an explanatory string to be passed to the
      ///                     std::runtime_error constructor
      /// \param simulation   a reference to the simulation being rejected
      Bad_Simulation_Error (std::string const& explanation, Simulation const& simulation):
        std::runtime_error(explanation), simulation_{simulation} {}

      /// Obtain a reference to the simulation being rejected.
      Simulation const& simulation() const noexcept {
        return simulation_;
      }

    private:

      std::reference_wrapper<Simulation const> simulation_;

  };

}

#endif
