# Delay Differential Equations Simulator
A generic library for simulating networks of ordinary and delay differential equations for systems modeling.

Contributors to this file should be aware of [Adam Pritchard's Markdown Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet).

## Table of Contents

| 0: [System Requirements](#0-system-requirements)  
|__ 0.0: [Operating System](#00-operating-system)  
|__ 0.1: [Compilers](#01-compilers)  
| 1: [Tutorial with Simple Model](#1-tutorial-with-simple-model)  
| 2: [Compilation and Model Building](#2-model-building)  
|__ 2.0: [Species and Reactions](#20-species-and-reactions)  
|____ 2.0.0: [Declaring Species](#200-declaring-species)  
|____ 2.0.1: [Declaring Reactions](#201-declaring-reactions)  
|____ 2.0.2: [Defining Reaction Formulas](#202-defining-reaction-formulas)  
|____ 2.0.3: [Defining Reaction Inputs and Outputs](#203-defining-reaction-inputs-and-outputs)  
|__ 2.1: [Compiling and Generating Parameter Templates](#21-compiling-and-generating-parameter-templates)  
|__ 2.2: [Parameters](#22-parameters)  
|____ 2.2.0: [CSV Parser Specifications](#220-csv-parser-specifications)  
|____ 2.2.1: [Parameter Sets](#221-parameter-sets)  
|____ 2.2.2: [Perturbations](#222-perturbations)  
|____ 2.2.3: [Gradients](#223-gradients)  
| 3: [Running the Simulation](#3-running-the-simulation)  
|__ 3.0: [Description of the Simulation](#30-description-of-the-simulation)  
|__ 3.1: [Input](#31-input)  
|____ 3.1.0: [Required Files](#310-required-files)  
|____ 3.1.1: [Optional Files](#311-optional-files)  
|____ 3.1.2: [Command Line Arguments](#312-command-line-arguments)  
|__ 3.2: [Output](#32-output)  
|____ 3.2.0: [Simulation Log](#320-simulation-log)  
|____ 3.2.1: [Analysis](#321-analysis)  
|______ 3.2.1.0: [Output Destination](#3210-output-destination)  
|______ 3.2.1.1: [Basic Analysis](#3211-basic-analysis)  
|______ 3.2.1.2: [Oscillation Analysis](#3212-oscillation-analysis)  
| 4: [Authorship and License](#4-authorship-and-license)  <<< PLACEHOLDER AUTHOR NAMES  

## 0: System Requirements

#### 0.0: Operating System

linux, mac. windows if-y

***
#### 0.1: Compilers

A C++ compiler with support for at least the C++11 standard is required. In order to compile the CUDA accelerated code, both a CUDA 6.0+ compiler and NVIDIA GPU hardware with "Compute Capability 3.0+" are needed.

[Back to Top](#delay-differential-equations-simulator)

## 1: Tutorial with Simple Model

step-by-step instructions on whole process using a simple model

[Back to Top](#delay-differential-equations-simulator)

## 2: Model Building

#### 2.0: Species and Reactions

***
#### 2.0.0: Declaring Species

Declare species in `specie_list.hpp`. List the specie names between the two sets of C++ macros (the lines that begin with `#`) in the same format as below. The following example lists two species, `alpha` and `bravo`, and one critical speice, `charlie`.

```
SPECIE(alpha)
SPECIE(bravo)
CRITICAL_SPECIE(charlie)
```

***
#### 2.0.1: Declaring Reactions

Declare reactions in `reactions_list.hpp`. List the reaction names between the two sets of C++ macros (the lines that begin with `#`) in the same format as below. The following example lists one delay reaction, `alpha_synthesis`, and three normal reactions, `bravo_synthesis`, `alpha_degredation`, and `bravo_degredation`. While this particular reaction naming scheme is not required, it can be helpful.

```
DELAY_REACTION(alpha_synthesis)
REACTION(bravo_synthesis)
REACTION(alpha_degredation)
REACTION(bravo_degredation)
```

***
#### 2.0.2: Defining Reaction Rate Formulas

Define all of reaction rate functions in `model_impl.hpp`.
For example, if a reaction is enumerated `R_ONE`, it should be declared as a 
   function like this:
```
 RATETYPE reaction<R_ONE>::active_rate(const Ctxt& c) const { return 6.0; }
```
 
Or, for a more interesting reaction rate, you might do something like:
 
```
 RATETYPE reaction<R_TWO>::active_rate(const Ctxt& c) const {
   return c.getRate(R_TWO) * c.getCon(SPECIE_ONE) * c.neighbors.calculateNeighborAvg(SPECIE_TWO);
 }
```
Refer to the Context API in the following section for instructions on how to get delays
   and critical values for more complex reaction rate functions.

***
### 2.0.3: Context API

Contexts are iterators over the concentration levels of all species in all cells. Use them to get the conc values of specific species that factor into reaction rate equations.

To get the concentration of a specie where `c` is the context object and `SPECIE` is the specie's enumeration:
`c.getCon(SPECIE)`

To get the delay time of a particular delay reaction that is enumerated as `R_ONE` and is properly identified as a delay reaction in `reactions_list.hpp` (see 2.0.1):
`RATETYPE delay_time = c.getDelay(dreact_R_ONE);`

To get the past concentration of `SPECIE` where `delay_time`, as specified in the previous example, is the delay time for `R_ONE`:
`c.getCon(SPECIE, delay_time);`

To get average concentration of SPECIE in that cell and its surrounding cells:
`c.calculateNeighborAvg(SPECIE)`

To get the past average concentration of SPECIE in that cell and its surround cells:
`c.calculateNeighborAvg(SPECIE, delay_time)`


***
#### 2.0.4: Defining Reaction Inputs and Outputs

Define each reaction's reactants and products in `reaction_deltas.hpp`.
Say a reaction enumerated as `R_ONE` has the following chemical formula:

                           2A + B --> C

The proper way to define that reaction's state change vector is as follows:
```
STATIC_VAR int num_deltas_R_ONE = 3;
STATIC_VAR int deltas_R_ONE[] = {-2, -1, 1};
STATIC_VAR specie_id delta_ids_R_ONE[] = {A, B, C};
```

***
#### 2.1: Compiling and Generating Parameter Templates

discuss how to cmake works, csv_gen dependency, and where the generated templates are

***
#### 2.2: Parameters

***
#### 2.2.0: CSV Parser Specifications

copy from and add extra to what is already in csv headers for each of these sections

***
#### 2.2.1: Parameter Sets

param_sets.csv

***
#### 2.2.2: Perturbations

param_pert.csv

***
#### 2.2.3: Gradients

param_grad.csv

[Back to Top](#delay-differential-equations-simulator)

## 3: Running the Simulation

#### 3.0: Description of Simulation

The application supports simulation of well-stirred chemical systems in a single enclosed environment and in multicellular networks.  The simulation uses delay-differential equations to advance a given model and estimate concentrations for given species updated by given reactions.  The size of *dt* varies depending on which simulation algorithm is run.

***
### 3.0.0: Deterministic Simulation

The Deterministic Simulation Algorithm uses rate reaction equations to approximate the concentration levels of every specie over a constant, user-specified, time step.  Files concerning this simulation type are found in `source/sim/determ`. The user can turn on deterministic simulation by specifying a time step in the command line (see 3.1.2).

***
### 3.0.1: Stochastic Simulation

The Stochastic Simulation Algorithm loosely follows Dan Gillespie's tau-leaping process.  The *dt* is calculated from a random variable as the time until the next reaction event occurs.  Molecular populations are treated as whole numbers and results are non-deterministic unless a random seed is provided by the user in the command line (see 3.1.2). The algorithm is much more performance intensive than the deterministic algorithm and is most ideal for smaller tissue sizes and shorter simulation durations.

***
#### 3.1: Input

***
#### 3.1.0: Required Files

param_sets.csv

***
#### 3.1.1: Optional Files

such as perturb and gradient files

***
#### 3.1.2: Command Line Arguments

big ol' list of args

***
#### 3.2: Output

***
#### 3.2.0: Simulation Log

how is data_out.csv (could be diff name, -e) formatted?

***
#### 3.2.1: Analysis

***
#### 3.2.1.0: Output Destination

as of now in cmd line, but could use " > anlys.txt" for example, or mod code, see 3.1.2.3

***
#### 3.2.1.1: Basic Analysis

Basic Analysis calculates the average concentration level of each specie over a given time interval for each cell of a given set and across all of the selected cells.  The object also calculates minimum and maximum concentration levels for each specie across the set of cells and for each cell.

***
#### 3.2.1.2: Oscillation Analysis

Oscillation Analysis identifies the local extrema of a given local range over a given time interval for a given set of cells.  The object also calculates the average period and amplitude of these oscillations for each cell in the given set.

[Back to Top](#delay-differential-equations-simulator)

***
### 3.2.1.3: Concentration Check

Concentration Check allows the user to abort simulation prematurely if a concentration level of a given specie (or for all species) escapes the bounds of a given lower and upper value for any given set of cells and time interval.

## 4: Authorship and License

Copyright (C) 2017 Billy Bob (bobb@example.edu), Joe Eskimo (eskimoj@example.edu), and Hal 9000 (9000h@example.edu).

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.

[Back to Top](#delay-differential-equations-simulator)
