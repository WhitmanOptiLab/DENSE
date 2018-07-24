#include "csvr.hpp"
#include "utility/style.hpp"
#include "utility/common_utils.hpp"

using style::Color;

#include <cfloat> // For FLT_MAX as an internal error code
#include <cmath>
#include <limits>
#include <iostream>


csvr::csvr(std::string const& file_name, bool suppress_file_not_found) :
    iFile{}, iLine(1) {

    auto file { std14::make_unique<std::ifstream>(file_name) };
    // Check if open successful
    if (!file->is_open() && !suppress_file_not_found) {
        std::cout << style::apply(Color::red) << "CSV file input failed. CSV file \'" <<
            file_name << "\' not found or open." << style::reset() << '\n';
        return;
    }

    iFile = std::move(file);
}

bool csvr::has_stream() const {
  return iFile.get();
}

bool csvr::skip_next() {
    return csvr::get_next(static_cast<Real *>(nullptr));
}

bool csvr::get_next(int* rate) {
  Real result;
  bool success = get_next(&result);
  if (success && rate) *rate = std::round(result);
  return success;
}

bool csvr::get_next(Real* pnRate) {
  return csvr::get_real(*iFile, pnRate);
}

bool csvr::get_real(std::istream& iFile, Real* pnRate) {
  // tParam data from file to be "pushed" to pfRate
  std::string tParam; int iLine = 0;

  char c;
  while (iFile >> c) {
      // Skip line comments
      if (c == '#') {
          iFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
          tParam.clear();
          ++iLine;
      }
      // Parse only if not whitespace except for \n
      else if (c != ' ' && c != '\t')
      {
        // Ignore non-numeric, non data seperator, non e, -, and + characters
        if (!( (c >= '0' && c <= '9') || c == '.' || c == ',' || c == '\n' ||
                c == 'e' || c == '-' || c == '+') )
        {
            iFile.ignore(std::numeric_limits<std::streamsize>::max(), ',');
            tParam.clear();
        }
        // If hit data seperator, add data to respective array
        // '\n' is there in case there is no comma after last item
        else if (c != ',' && c != '\n') {
          tParam += c;
        }
        else if (!tParam.empty()) {
          char* tInvalidAt;
          Real tRate = strtold(tParam.c_str(), &tInvalidAt);

          // If found invalid while parsing
          if (*tInvalidAt)
          {
              std::cout << style::apply(Color::red) <<
                  "CSV parsing failed. Invalid data contained "
                  "at line " << "(unknown)" << "." <<
                  style::reset() << '\n';
              return false;
          }
          // Else was success
          if (pnRate) { *pnRate = tRate; }
          return true;
        }
      }

      // Increment line counter
      if (c == '\n') {
          ++iLine;
      }
  }

  // End of file
  return false;
}
