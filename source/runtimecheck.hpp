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
        double duration;
        std::chrono::high_resolution_clock::time_point begin;
        std::chrono::high_resolution_clock::time_point end;
    
        std::vector<double> durations;
        std::vector<std::chrono::high_resolution_clock::time_point> beginnings;
        std::vector<std::chrono::high_resolution_clock::time_point> endings;
    public:
        runtimecheck(){
            begin = std::chrono::high_resolution_clock::now();
            beginnings.emplace_back(begin);
        }
    
        void set_end(){
            end = std::chrono::high_resolution_clock::now();
            endings.emplace_back(end);
        }
    
        void set_begin(){
            begin = std::chrono::high_resolution_clock::now();
            beginnings.emplace_back(begin);
        }
    
        double get_duration(int i, int j){
            duration = std::chrono::duration_cast<std::chrono::microseconds>( endings[i] - beginnings[j] ).count();
            durations.emplace_back(duration);
            return duration;
        }
    
        /*std::vector<double> duration_list (){
            return durations;
        }*/
    
        virtual ~runtimecheck() = default;
        runtimecheck(runtimecheck&&)  = default;
};

#endif /* runtimecheck_h */
