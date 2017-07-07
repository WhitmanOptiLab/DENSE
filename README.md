# Delay Differential Equations Simulator
A generic library for simulating networks of ordinary and delay differential equations for systems modeling

## Table of Contents

| 0: [System Requirements](#0-system-requirements)  
|__ 0.0: [Operating System](#00-operating-system)  
|__ 0.1: [Compilers](#01-compilers)  
| 1: [Model Building](#1-model-building)  
|__ 1.0: [Species and Reactions](#10-species-and-reactions)  
|____ 1.0.0: [Declaring Species](#100-declaring-species)  
|____ 1.0.1: Declaring Reactions  
|____ 1.0.2: Defining Reaction Formulas  
|____ 1.0.3: Defining Reaction Inputs and Outputs  
|__ 1.1: Generating Parameter File Templates {no discussion of how cmake works yet, just tell them how to use}  
|__ 1.2: Parameters
|____ 1.2.0: CSV Parser Specifications  {mostly what is just in the csv headers}  
|____ 1.2.1: Parameter Sets {param_sets.csv}  
|____ 1.2.2: Perturbations {param_pert.csv}  
|____ 1.2.3: Gradients {param_grad.csv}  
| 2: [Compilation](#2-compilation)  
|__ 2.0: {}

This chunck was copied over from sogen_2014 -- replace later keeping in mind that this is a DDE simulator, not a biological simulator
| 3: Running the Simulation  
|__ 3.0: Biological and computational description of the simulation  
|__ 3.1: Setting up a simulation  
|__ 3.2: Input and output  
|____ 3.2.0: Methods of input  
|____ 3.2.1: Parameter sets  
|____ 3.2.2: Perturbations and gradients  
|____ 3.2.3: Command-line arguments  
|____ 3.2.4: Input file formats  
|______ 3.2.4.0: Parameter sets format  
|______ 3.2.4.1: Perturbations format  
|______ 3.2.4.2: Gradients format  
|______ 3.2.4.3: Ranges format  
|____ 3.2.5: Output file formats  
|______ 3.2.5.0: Passed sets format  
|______ 3.2.5.1: Concentrations text format  
|______ 3.2.5.2: Concentrations binary format  
|______ 3.2.5.3: Oscillation features format  
|______ 3.2.5.4: Conditions format  
|______ 3.2.5.5: Scores format  
|______ 3.2.5.6: Seeds format  
|____ 3.2.6: Piping in parameter sets from other applications  
|____ 3.2.7: Generating random parameter sets  
|__ 3.3: Modifying the code  
|____ 3.3.0: Adding command-line arguments  
|____ 3.3.1: Adding input and output files  
|______ 3.3.1.0: Input files  
|______ 3.3.1.1: Output files  
|____ 3.3.2: Adding mutants  
|____ 3.3.3: Adding genes  
|____ 3.3.5: Other modifications  

| 4: Analysis 
|__ 4.0:  
|____ 4.0.0:  
|____ 4.0.1:  
| 4: License

## 0: System Requirements

### 0.0: Operating System

linux only?

***

### 0.1: Compilers

g++, cuda (and which generations of nvidia gpu?)

***

## 1: Model Building

***

### 1.0: Species and Reactions

***

### 1.0.0: Declaring Species

{how to do specie_list.hpp}

***

### 1.0.1: Declaring Reactions

{how to do reactions_list.hpp}

***

### 1.0.2: Defining Reaction Formulas

{how to do model_impl.hpp}

***

### 1.0.3: Defining Reaction Inputs and Outputs

{how to do reaction.cpp/.cu -- though this will change a lot depending on how we resulve issue #11}

