# Delay Differential Equations Simulator
A generic library for simulating networks of ordinary and delay differential equations for systems modeling

## Table of Contents

| 0: [System Requirements](#0-system-requirements)  
|__ 0.0: [Operating System](#00-operating-system)  
|__ 0.1: [Compilers](#01-compilers)  
| 1: [Compilation and Model Building](#1-model-building)  
|__ 1.0: [Species and Reactions](#10-species-and-reactions)  
|____ 1.0.0: [Declaring Species](#100-declaring-species)  
|____ 1.0.1: [Declaring Reactions](#101-declaring-reactions)  
|____ 1.0.2: [Defining Reaction Formulas](#102-defining-reaction-formulas)  
|____ 1.0.3: [Defining Reaction Inputs and Outputs](#103-defining-reaction-inputs-and-outputs)  
|__ 1.1: [Compiling and Generating Parameter Templates](#11-compiling-and-generating-parameter-templates)  
|__ 1.2: [Parameters](#12-parameters)  
|____ 1.2.0: [CSV Parser Specifications](#120-csv-parser-specifications)  
|____ 1.2.1: [Parameter Sets](#121-parameter-sets)  
|____ 1.2.2: [Perturbations](#122-perturbations)  
|____ 1.2.3: [Gradients](#123-gradients)  
| 2: [Running the Simulation](#2-running-the-simulation)  
|__ 2.0: [Description of the Simulation](#20-description-of-the-simulation)  
|__ 2.1: Input  
|____ 2.1.0: Required Files
|____ 2.1.1: Optional Files
|____ 2.1.2: Command Line Arguments  
|__ 2.2: Output  
|____ 2.2.0: Simulation Log  
|____ 2.2.1: Analysis  
|______ 2.2.1.0: Output Destination
|______ 2.2.1.1: Basic Analysis  
|______ 2.2.1.2: Oscillation Analysis  
| 3: Source Code  
|__ 3.0: Organization  
|____ 3.0.0: Principle Paradigms  
|____ 3.0.n: for each folder in source (see issue #20) describe its purpose  
|__ 3.1: Modding  
|____ 3.1.0: Algorithms  
|______ 3.1.0.0: Simulation  
|______ 3.1.0.1: Analysis  
|____ 3.1.1: Model  
|____ 3.1.2: I/O
|______ 3.1.2.0: Command Line Arguments  
|______ 3.1.2.1: Input C++ Files  
|______ 3.1.2.2: Input CSV Files
|______ 3.1.2.3: Output CSV Files  
| 4: Documentation?  
|__ 4.n: for each folder  
|____ 4.n.n: for each file and class  
|______ 4.n.n.n: for each function and field  
| 5: Authorship and License  

## 0: System Requirements

#### 0.0: Operating System

linux only?

***
#### 0.1: Compilers

g++, cuda (and which generations of nvidia gpu?)

[Back to Top](#delay-differential-equations-simulator)

## 1: Model Building

#### 1.0: Species and Reactions

***
#### 1.0.0: Declaring Species

how to do specie_list.hpp

***
#### 1.0.1: Declaring Reactions

how to do reactions_list.hpp

***
#### 1.0.2: Defining Reaction Formulas

how to do model_impl.hpp

***
#### 1.0.3: Defining Reaction Inputs and Outputs

how to do reaction.cpp/.cu -- though this will change a lot depending on how we resulve issue #11

***
#### 1.1: Compiling and Generating Parameter Templates

discuss how to cmake works, csv_gen dependency, and where the generated templates are

***
#### 1.2: Parameters

***
#### 1.2.0: CSV Parser Specifications

copy from and add extra to what is already in csv headers for each of these sections

***
#### 1.2.1: Parameter Sets

param_sets.csv

***
#### 1.2.2: Perturbations

param_pert.csv

***
#### 1.2.3: Gradients

param_grad.csv

[Back to Top](#delay-differential-equations-simulator)

## 2: Running the Simulation

#### 2.0: Description of the Simulation

mind that this is a differential equations simulator, not just a biology simulator

***
#### 2.1: Input

***
#### 2.1.0: Required Files

such as all in /models/blahblahblah

***
#### 2.2.1: Analysis

***
#### 2.2.1.0: Output Destination

as of now in cmd line, but could use " > out.dat" for ex or mod code, see 3.1.5

***
#### 2.2.1.1: Basic Analysis

what does it tell us?

***
#### 2.2.1.2: Oscillation Analysis

what's its format?

[Back to Top](#delay-differential-equations-simulator)

## 3: Source Code

#### 3.0: Organization

***
#### 3.0.0: Principle Paradigms

such as observer-observable, inheritance of simulation_base, etc

***
#### 3.0.n: folder n

describe folder n's purpose

***
#### 3.1: Modding

***
#### 3.1.0: Algorithms

***
#### 3.1.0.0: Simulation

how to add new simulation class + implementation of context

***
#### 3.1.0.1: Analysis

using observer interface and/or analysis inheritance to do analyses

***
#### 3.1.1: Model

go back to section 1. RECOMPILE IS NECESSARY!!! reiterate that each compile is model-specific

***
#### 3.1.2: I/O

***
#### 3.1.2.0: Command Line Arguments

how to add/remove. how they're designed. intended design pattern/usage

***
#### 3.1.2.1: Input C++ Files

***
#### 3.1.2.2: Input CSV Files

***
#### 3.1.2.3: Output CSV Files

[Back to Top](#delay-differential-equations-simulator)

## 4: Documentation

do we need this section?

## 5: Authorship and License

Copyright (C) John Smith (smithj@example.edu), John Miller (millerj@example.edu), and John Carpenter (carpenterj@example.edu).

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.

[Back to Top](#delay-differential-equations-simulator)
