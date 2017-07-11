# Delay Differential Equations Simulator Documentation

| 0: [Principle Paradigms](#0-principle-paradigms)  
| 1: [Organization](#1-organization)  
|__ 1.0: [/anlys](#10-anlys)  
|__ 1.1: [/core](#11-model)  
|__ 1.2: [/io](#12-io)  
|__ 1.3: [/sim](#13-sim)  
|__ 1.4: [/util](#14-util)  
| 2: [Modding](#2-modding)  
|__ 2.0: [Algorithms](#20-algorithms)  
|____ 2.0.0: [Simulation](#200-simulation)  
|____ 2.0.1: [Analysis](#201-analysis)  
|__ 2.1: [Model](#21-model)  
|__ 2.2: [I/O](#22-io)  
|____ 2.2.0: [Command Line Arguments](#220-command-line-arguments)  
|____ 2.2.1: [Input C++ and Header Files](#221-input-c-and-header-files)  
|____ 2.2.2: [Input CSV Files](#222-input-csv-files)  
|____ 2.2.3: [Output CSV Files](#223-output-csv-files)  
| 3: [Documentation](#3-documentation) ?  
|__ 3.n: for each folder  
|____ 3.n.n: for each file and class  
|______ 3.n.n.n: for each function and field  

## 0: Principle Paradigms

such as observer-observable, inheritance of simulation_base, etc

[Back to Top](#delay-differential-equations-simulator-documentation)

## 1: Organization

#### 1.0: /anlys

* only /sim can access /anlys
* /anlys can access everyone except /sim

***
#### 1.1: /core

* everyone except /util can access /core
* /core can only access /io and /util

***
#### 1.2: /io

* only /anlys and /sim can access /io
* /io can only access /core and /util

***
#### 1.3: /sim

no one can access /sim
/sim can access everyone

***
#### 1.4: /util

* everyone can access /util
* /util cannot access anyone

[Back to Top](#delay-differential-equations-simulator-documentation)

## 2: Modding

#### 2.0: Algorithms

***
#### 2.0.0: Simulation

how to add new simulation class + implementation of context

***
#### 2.0.1: Analysis

using observer interface and/or analysis inheritance to do analyses

***
#### 2.1: Model

see section 2 of 'README.md'. RECOMPILE IS NECESSARY!!! reiterate that each compile is model-specific

***
#### 2.2: I/O

***
#### 2.2.0: Command Line Arguments

how to add/remove. how they're designed. intended design pattern/usage

***
#### 2.2.1: Input C++ and Header Files

add these to models folder, mention xmacros and suggest resources for learning them

***
#### 2.2.2: Input CSV Files

csvr description

***
#### 2.2.3: Output CSV Files

csvw description

[Back to Top](#delay-differential-equations-simulator-documentation)

## 3: Documentation

do we need this section?

