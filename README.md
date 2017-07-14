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

NIKI

***
#### 2.0.3: Defining Reaction Inputs and Outputs

NIKI

***
#### 2.1: Compiling and Generating Parameter Templates

Running `make` after having initialized CMake in the desired directory will automatically run `csv_gen` as the simulation is being compiled. `csv_gen` will generate `*_template.csv` files formatted for the directory's particular model. The easiest way to fill these out is with an Excel-like program such as LibreOffice Calc. Remember to always save changes using the original `*.csv` file extension. Changes should also be saved in a file name different from the one automatically generated so that there is no chance `csv_gen` will overwrite your settings.

***
#### 2.2: Parameters

***
#### 2.2.0: CSV Parser Specifications

At its core, CSV files contain numerical values seperated by commas. Below are a few extra rules that the simulation's CSV parser follows:
1. Empty cells, blank rows, and whitespace

   To illustrate, the following two examples are equivalent.
   Example A:
   ```
   3.14, , 2001, 2.18,
   
   41,       2.22e-22
   ```
   Example B:
   ```
   3.14,2001,2.18,41,2.22e-22
   ```

2. Comments, i.e. rows that begin with a `#`

...```
...# I am a comment! Below is the data.
...9182, 667
...```

3. Any cell that contains a character which is not a number, `.`, `+`, `-`, or `e`.

...Often times cells which do not contain numbers are intended to be column headers. These are not parsed by the simulation, and can technically be modified by the users as they wish.
...```
...alpha_synthesis, bravo_synthesis, charlie_synthesis, 
...21.12021,        33,              101.123, 
...```

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

#### 3.0: Description of the Simulation

mind that this is a differential equations simulator, not just a biology simulator

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

TODO LATER... will change next week based on what we do with analysis log

***
#### 3.2.1.1: Basic Analysis

NIKI

***
#### 3.2.1.2: Oscillation Analysis

NIKI

[Back to Top](#delay-differential-equations-simulator)

## 4: Authorship and License

Copyright (C) 2017 Billy Bob (bobb@example.edu), Joe Eskimo (eskimoj@example.edu), and Hal 9000 (9000h@example.edu).

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.

[Back to Top](#delay-differential-equations-simulator)
