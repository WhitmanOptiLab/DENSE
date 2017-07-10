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

This chunck was copied over from sogen_2014 -- replace later keeping in mind that this is a DDE simulator, not a biological simulator  
| 2: Running the Simulation  
|__ 2.0: [Description of the Simulation](#20-description-of-the-simulation)  
|__ 2.1: Input and output  
|____ 2.1.0: Methods of input  
|____ 2.2.1: Parameter sets  
|____ 2.2.2: Perturbations and gradients  
|____ 2.2.3: Command-line arguments  
|____ 2.2.4: Input file formats  
|______ 2.2.4.0: Parameter sets format  
|______ 2.2.4.1: Perturbations format  
|______ 2.2.4.2: Gradients format  
|______ 2.2.4.3: Ranges format  
|____ 2.2.5: Output file formats  
|______ 2.2.5.0: Passed sets format  
|______ 2.2.5.1: Concentrations text format  
|______ 2.2.5.2: Concentrations binary format  
|______ 2.2.5.3: Oscillation features format  
|______ 2.2.5.4: Conditions format  
|______ 2.2.5.5: Scores format  
|______ 2.2.5.6: Seeds format  
|____ 2.2.6: Piping in parameter sets from other applications  
|____ 2.2.7: Generating random parameter sets  
|__ 2.3: Modifying the code  
|____ 2.3.0: Adding command-line arguments  
|____ 2.3.1: Adding input and output files  
|______ 2.3.1.0: Input files  
|______ 2.3.1.1: Output files  
|____ 2.3.2: Adding mutants  
|____ 2.3.3: Adding genes  
|____ 2.3.5: Other modifications  

| 3: Analysis  
|__ 3.0:  
|____ 3.0.0:  
|____ 3.0.1:  
| 4: Source Code Overview  
|__ 4.0: Organization  
|____ 4.0.n: for each folder in source (see issue #20) describe its purpose  
|__ 4.1: Modding  
|____ 4.1.0: Adding Simulation Algorithms  
|____ 4.1.1: Adding Different Analyses  
| 5: Documentation?  
|__ 5.n: for each folder  
|____ 5.n.n: for each file and class  
|______ 5.n.n.n: for each function and field  
| 6: License  

## 0: System Requirements

#### 0.0: Operating System

linux only?

***
#### 0.1: Compilers

g++, cuda (and which generations of nvidia gpu?)

***
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

***
[Back to Top](#delay-differential-equations-simulator)

## 2: Running the Simulation

