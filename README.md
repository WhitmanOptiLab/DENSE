# DENSE: <sub>a Differential Equation Network Simulation Engine</sub>
A generic library for simulating networks of ordinary and delay differential equations for systems modeling.

Contributors to this file should be aware of [Adam Pritchard's Markdown Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet).

## Table of Contents

0. [System Requirements](#0-system-requirements)
    0. [Operating System](#00-operating-system)
    1. [Compilers](#01-compilers)
1. [Tutorial with Simple Model](#1-tutorial-with-simple-model)
2. [Compilation and Model Building](#2-model-building)
    0. [Species and Reactions](#20-species-and-reactions)
        0. [Declaring Species](#200-declaring-species)
        1. [Declaring Reactions](#201-declaring-reactions)
        2. [Defining Reaction Rate Formulas](#202-defining-reaction-rate-formulas)
        3. [Defining Reaction Inputs and Outputs](#203-defining-reaction-inputs-and-outputs)
    1. [Compiling and Generating Parameter Templates](#21-compiling-and-generating-parameter-templates)
    2. [Parameters](#22-parameters)
        0. [CSV Parser Specifications](#220-csv-parser-specifications)
        1. [Parameter Sets](#221-parameter-sets)
        2. [Perturbations](#222-perturbations)
        3. [Gradients](#223-gradients)
        4. [Anyses XML](#224-analyses-xml)
3. [Running the Simulation](#3-running-the-simulation)
    0. [Description of the Simulation](#30-description-of-the-simulation)
        0. [Preamble](#300-preamble)
        1. [Deterministic](#301-deterministic)
        2. [Stochastic](#302-stochastic)
    1. [Input](#31-input)
        0. [Required Files](#310-required-files)
        1. [Optional Files](#311-optional-files)
        2. [Command Line Arguments](#312-command-line-arguments)
    2. [Output](#32-output)
        0. [Simulation Log](#320-simulation-log)
        1. [Analysis](#321-analysis)
        0. [Output Destination](#3210-output-destination)
        1. [Basic Analysis](#3211-basic-analysis)
        2. [Oscillation Analysis](#3212-oscillation-analysis)
        3. [Concentration Check](#3213-concentration-check)
4. [Authorship and License](#4-authorship-and-license)

## 0: System Requirements

#### 0.0: Operating System

linux, mac. windows if-y

***
#### 0.1: Compilers

CMake version 2.8+ required, along with a supported build manager (make is the only tested system currently.)
A C++ compiler with support for at least the C++11 standard is required.
In order to compile the CUDA accelerated code, both a CUDA 6.0+ compiler and NVIDIA GPU hardware with "Compute Capability 3.0+" are needed.

[Back to Top](#delay-differential-equations-simulator)

## 1: Tutorial with Simple Model

step-by-step instructions on whole process using a simple model

[Back to Top](#delay-differential-equations-simulator)

## 2: Model Building

#### 2.0: Species and Reactions

***
#### 2.0.0: Declaring Species

Declare species in `specie_list.hpp`. List the specie names between the two sets of C++ macros (the lines that begin with `#`) in the same format as below. The following example lists two normal species, `alpha` and `charlie`, and one critical specie, `bravo`.

A critical specie is a specie whose effect on a particular reaction is bounded by a threshhold. Imagine a specie that doesn't have an effect on a reaction rate until it reaches a certain concentration or might stop having a linear relationship with the rate after a certain conc level.  Whether this critical value is an upper or lower bound is decided in the rate equations in `model_impl.hpp` but the value itself is inputted alongside delays and initial rates (see 3.1) in the parameter set. To establish the pairing of this specie and its critical value, it is declared as a critical specie.

<!---
[comment]: # What is a critical specie?
[comment]: # Do the critical species have to come last?  At a high level, what does this declaration do?
--->

```
SPECIE(alpha)
CRITICAL_SPECIE(bravo)
SPECIE(charlie)
```

***
#### 2.0.1: Declaring Reactions

Declare reactions in `reactions_list.hpp`. List the reaction names between the two sets of C++ macros (the lines that begin with `#`) in the same format as below. The following example lists one delay reaction, `alpha_degredation`, and three normal reactions, `bravo_synthesis`, `alpha_synthesis`, and `bravo_degredation`. While this particular reaction naming scheme is not required, it can be helpful.

<!---
[comment]: # Do the delay reactions have to come first?  At a high level, what does this declaration do?
--->

```
REACTION(alpha_synthesis)
REACTION(bravo_synthesis)
DELAY_REACTION(alpha_degredation)
REACTION(bravo_degredation)
```

***
#### 2.0.2: Defining Reaction Rate Formulas

<!---
[comment]: # keep the enumerations consistent, demonstrate that the active rate functions are tied to the previously declared reactions
--->

Define all of reaction rate functions in `model_impl.hpp`.
For example, if a reaction is enumerated `alpha_synthesis`, it should be declared as a
   function like this:
```
 RATETYPE reaction<alpha_synthesis>::active_rate(const Ctxt& c) const { return 6.0; }
```

Or, for a more interesting reaction rate, you might do something like:

```
 RATETYPE reaction<bravo_degredation>::active_rate(const Ctxt& c) const {
   return c.getRate(bravo_degredation) * c.getCon(bravo) * c.neighbors.calculateNeighborAvg(charlie);
 }
```
Refer to the Context API in the following section for instructions on how to get delays
   and critical values for more complex reaction rate functions.

***
#### 2.0.3: Context API

Contexts are iterators over the concentration levels of all species in all cells. Use them to get the conc values of specific species that factor into reaction rate equations.

To get the concentration of a specie where `c` is the context object and `bravo` is the specie's enumeration:
```c.getCon(bravo)```

To get the delay time of a particular delay reaction that is enumerated as `alpha_degredation` and is properly identified as a delay reaction in `reactions_list.hpp` (see [2.0.1: Declaring Reactions](#201-declaring-reactions)):
```RATETYPE delay_time = c.getDelay(dreact_alpha_degredation);```

To get the past concentration of `alpha` where `delay_time`, as specified in the previous example, is the delay time for `R_ONE`:
```c.getCon(alpha, delay_time);```

To get average concentration of charlie in that cell and its surrounding cells:
```c.calculateNeighborAvg(charlie)```

To get the past average concentration of SPECIE in that cell and its surround cells:
```c.calculateNeighborAvg(charlie, delay_time)```

To get the critical value of a critical specie that is enumerated as `bravo` and is properly identified as a critical specie in `specie_list.hpp` (see [2.0.0: Declaring Species](#200-declaring-species)):
```c.getCritVal(rcrit_bravo)```


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

Running `make` after having initialized CMake in the desired directory will automatically run `csv_gen` as the simulation is being compiled. `csv_gen` will generate `*_template.csv` files formatted for the directory's particular model. The easiest way to fill these out is with an Excel-like program such as LibreOffice Calc. Remember to always save changes using the original `*.csv` file extension. Changes should also be saved in a file name different from the one automatically generated so that there is no chance `csv_gen` will overwrite your settings.

***
#### 2.2: Parameters

<!---
[comment]: # Add something here to talk about how the model declaration informs the parameter sets.  Highlight that all reactions have rate constants, delay reactions have delays, and critical species have critical values.
--->

***
#### 2.2.0: CSV Parser Specifications

At its core, CSV files contain numerical values seperated by commas. Listed below are three categories of characters/strings that the simulation's CSV parser __*DOES NOT*__ parse.
1. Empty cells, blank rows, and whitespace

   To illustrate, the following two examples are equivalent.
   ```
   3.14, , 2001, -2.18,

   41,       2.22e-22
   ```
   ```
   3.14,2001,-2.18,41,2.22e-22
   ```

2. Comments, i.e. rows that begin with a `#`

   ```
   # I am a comment! Below is the data.
   9182, 667
   ```

3. Any cell that contains a character which is not a number, `.`, `+`, `-`, or `e`.

   Only the following scientific notation is supported:
   ```
   -0.314e+1, 3.00e+8, 6.63e-34
   ```
   These would be considered invalid:
   ```
   3.33*10^4, 1.3E-12, 4.4x10^2
   ```
   Often times cells which do not contain numbers are intended to be column headers. These are not parsed by the simulation, and can technically be modified by users as they wish.
   ```
   flying_pig_synthesis, plutonium_synthesis, cultural_degredation,
   21.12021,             33,                  101.123,
   ```
   It is futile, however, to add/remove/modify the column headers with the expectation of changing the program's behavior. Data must be entered in the default order if it is to be parsed properly.

***
#### 2.2.1: Parameter Sets

The parameter set template is named `param_sets_template.csv` by default. Parameter set files can contain more than one set per file (each being on their own line). When a file is loaded into the simulation, all sets are initialized and executed in parallel.

Below is an example of a parameter sets file that contains three sets:
```
alpha_synthesis, alpha_degredation,
332,             101,
301,             120,
9.99e+99,        1.0e-99,
```

***
#### 2.2.2: Perturbations

The perturbations template is named `param_pert_template.csv` by default. Perturbation files should only contain one set of perturbations. Only this one perturbations set is applied to all parameter sets when a simulation set is being run.

Use `0` to indicate that a reaction should not have perturbations. In the example below, `alpha_synthesis` has a perturbation while `alpha_degredation` does not.
```
alpha_synthesis, alpha_degredation,
0.05,            0,
```

***
#### 2.2.3: Gradients

The gradients template is named `param_grad_template.csv` by default. Gradient files should only contain one set of gradients. Only this one gradients set is applied to all parameter sets when a simulation set is being run.

Use `0` under all four columns of a reaction to indicate that it should not have a gradient. In the first example, `alpha_synthesis` does not have a gradient, while in the second example, `alpha_degredation` does.
```
alpha_synthesis_x1, alpha_synthesis_y1, alpha_synthesis_x2, alpha_synthesis_y2,
0,                  0,                  0,                  0,
```
```
alpha_synthesis_x1, alpha_synthesis_y1, alpha_synthesis_x2, alpha_synthesis_y2,
2,                  0.8,                5,                  2.52,
```
`alpha_degredation`'s gradient is between cell columns 2 and 5, with a multiplier of 0.8 starting at column 2, linearly increasing to 2.52 by column 5.

Gradient Suffixes Chart

| Suffix | Meaning          |
| ------ | ---------------- |
| x1     | start column     |
| y1     | start multiplier |
| x2     | end column       |
| y2     | end multiplier   |

***
#### 2.2.4: Analyses XML

add instructions here on `analyses.xml`

[Back to Top](#delay-differential-equations-simulator)

## 3: Running the Simulation

#### 3.0: Description of the Simulation

***
#### 3.0.0: Preamble

The application supports simulation of well-stirred chemical systems in a single enclosed environment and in multicellular networks.  The simulation uses delay-differential equations to advance a given model and estimate concentrations for given species updated by given reactions.  The size of *dt* varies depending on which simulation algorithm is run.

***
#### 3.0.1: Deterministic

The Deterministic Simulation Algorithm uses rate reaction equations to approximate the concentration levels of every specie over a constant, user-specified, time step.  Files concerning this simulation type are found in `source/sim/determ`. The user can turn on deterministic simulation by specifying a time step in the command line (see 3.1.2).

***
#### 3.0.2: Stochastic

The Stochastic Simulation Algorithm loosely follows Dan Gillespie's tau-leaping process.  The *dt* is calculated from a random variable as the time until the next reaction event occurs.  Molecular populations are treated as whole numbers and results are non-deterministic unless a random seed is provided by the user in the command line (see 3.1.2). The algorithm is much more performance intensive than the deterministic algorithm and is most ideal for smaller tissue sizes and shorter simulation durations.

***
#### 3.1: Input

***
#### 3.1.0: Required Files

After the simulation has been compiled, the only file required to run a deterministic or stochastic simulation is a filled out `param_sets_template.csv`. It is suggested that this file be renamed to `param_sets.csv` upon completion.

***
#### 3.1.1: Optional Files

In order to take advantage of perturbations and gradients, `param_pert_template.csv` and `param_grad_template.csv` need to be filled out. Rename these to `param_pert.csv` and `param_grad.csv` or something similar upon completion.

Similarly, analyses must also be configured in a seperate file. A blank `analyses.xml` should be included in the `/template_model` directory.

***
#### 3.1.2: Command Line Arguments

The table below can also be accessed by running `simulation` either without any command line arguments or with any of the following flags: `-h`, `--help`, `--usage`.
Short and long flags are equivalent; either can be used to get the same program behavior. Short flags must be preceeded by `-` while long flags must be preceeded by `--`. As examples: `-h` and `--help`. Arguments that have a parameter require additional text to proceed the flag itself like so: `-p ../param_sets.csv`, `-b 0.05`, and `-o "alpha, bravo"`.

`RATETYPE` is set to `double` (double-precision floating-point) by default and can be changed in `source/util/common_utils.hpp`.

| Short | Long              | Parameter  | Description
| ----- | ----------------- | ---------- | -----------
| `h`   | `help` or `usage` | *none*     | Print information about all command line arguments.
| `n`   | `no-color`        | *none*     | Disable color in the terminal.
| `p`   | `param-sets`      | `string`   | Relative file location and name of the parameter sets `*.csv`. `../param_sets.csv`, for example.
| `g`   | `gradients`       | `string`   | Enables gradients and specifies the relative file location and name of the gradients `*.csv`. `../param_grad.csv`, for example.
| `b`   | `perturbations`   | `string`   | Enables perturbations and specifies the relative file location and name of the perturbations `*.csv`. `../param_pert.csv`, for example.
| `b`   | `perturbations`   | `RATETYPE` | Enables perturbations and specifices a global perturbation factor to be applied to ALL reactions. The `-b | --perturb` flag itself is identical to the `string` version; the simulation automatically detects whether it is in the format of a file or `RATETYPE`.
| `e`   | `data-export`     | `string`   | Relative file location and name of the output of the logged data `*.csv`. `../data_out.csv`, for example.
| `o`   | `specie-option`   | `string`   | Specify which species the logged data csv should output. This argument is only useful if `-e | --data-export` is being used. Not including this argument, or entering `all` for the parameter, makes the program by default output/analyze all species. __*IF MORE THAN ONE SPECIE, BUT NOT ALL, IS DESIRED*__, enclose the argument in quotation marks and seperate the species using commas. For example, `-o "alpha, bravo, charlie"`. If only one specie is desired, no commas or quotation marks are necessary.
| `i`   | `data-import`     | `string`   | Relative file location and name of `*.csv` data to import into the analyses. `../data_in.csv`, for example. Using this flag runs only analysis.
| `a`   | `analysis`     | `string`   | Relative file location and name of the analysis config `*.xml` file. `../analyses.xml`, for example. USING THIS ARGUMENT IMPLICITLY TOGGLES ANALYSIS.
| `c`   | `cell-total`      | `int`      | Total number of cells to simulate.
| `w`   | `total-width`     | `int`      | Width of tissue to simulate. Height is inferred by dividing `cell-total` by `total-width`.
| `r`   | `rand-seed`       | `int`      | Set the stochastic simulation random number generator seed.
| `s`   | `step-size`       | `RATETYPE` | Time increment by which the deterministic simulation progresses. __*USING THIS ARGUMENT IMPLICITLY SWITCHES THE SIMULATION FROM STOCHASTIC TO DETERMINISTIC*__.
| `t`   | `time-total`      | `RATETYPE` | Amount of simulated minutes the simulation should execute.
| `u`   | `anlys-intvl`     | `RATETYPE` | Analysis __*AND*__ file writing interval. How frequently (in units of simulated minutes) data is fetched from simulation for analysis and/or file writing.
| `v`   | `time-col`        | *none*     | Toggles whether file output includes a time column. Convenient for making graphs in Excel-like programs but slows down file writing. Time could be inferred without this column through the row number and the analysis interval.

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

Basic Analysis calculates the average concentration level of each specie over a given time interval for each cell of a given set and across all of the selected cells.  The object also calculates minimum and maximum concentration levels for each specie across the set of cells and for each cell.

MAKE EDITS DEPENDING ON IMPLEMENTATION OF ANALYSIS LOG

***
#### 3.2.1.2: Oscillation Analysis

Oscillation Analysis identifies the local extrema of a given local range over a given time interval for a given set of cells. The object also calculates the average period and amplitude of these oscillations for each cell in the given set.
MAKE EDITS DEPENDING ON IMPLEMENTATION OF ANALYSIS LOG
***
#### 3.2.1.3: Concentration Check

Concentration Check allows the user to abort simulation prematurely if a concentration level of a given specie (or for all species) escapes the bounds of a given lower and upper value for any given set of cells and time interval.

[Back to Top](#delay-differential-equations-simulator)

## 4: Authorship and License

Copyright (C) 2016-2017 John Stratton (strattja@whitman.edu), Ahmet Ay (aay@colgate.edu)

This software was developed with significant contributions from several students over the years: Yecheng Yang, Nikhil Lonberg, and Kirk Lange.

Copying and distribution of this file, with or without modification, are permitted in any medium without royalty provided the copyright notice and this notice are preserved.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

[Back to Top](#delay-differential-equations-simulator)
