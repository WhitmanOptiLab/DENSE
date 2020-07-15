//
//  runtimecheck.hpp
//  
//
//  Created by Myan Sudharsanan on 7/14/20.
//

#ifndef runtimecheck_h
#define runtimecheck_h
#include <chrono>
#include <cstdlib>
#include <cassert>
#include <random>
#include <memory>
#include <iterator>
#include <algorithm>
#include <functional>
#include <exception>
#include <iostream>
#include <utility>
#include <map>

class runtimecheck{
    private:
        auto duration;
        auto begin;
        auto end;
    
        std::vector<auto> durations;
        std::vector<auto> beginnings;
        std::vector<auto> endings;
    public:
        runtimecheck(){
            begin = std::chrono::high_resolution_clock::now();
            beginnings.push_back(begin);
        }
    
        void set_end(){
            end = std::chrono::high_resolution_clock::now();
            endings.push_back(end);
        }
    
        void set_begin(){
            begin = std::chrono::high_resolution_clock::now();
            beginnings.push_back(begin);
        }
    
        auto get_duration(int i, int j){
            duration = std::chrono::duration_cast<std::chrono::microseconds>( end[i] - begin[j] ).count();
            durations.push_back(duration);
            return duration;
        }
    
        std::vector<auto> duration_list (){
            return durations;
        }
    
        virtual ~runtimecheck() = default;
        runtimecheck(runtimecheck&&)  = default;
};

#endif /* runtimecheck_h */
