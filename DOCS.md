# DDE Documentation
Notes on the design principles, organization, and modability of the DDE simulation

## Table of Contents

| 0: [Principle Paradigms](#0-principle-paradigms)  
| 1: [Organization](#1-organization)  
|__ 1.0: [/anlys](#10-anlys)  
|__ 1.1: [/core](#11-model)  
|__ 1.2: [/io](#12-io)  
|__ 1.3: [/sim](#13-sim)  
|__ 1.4: [/util](#14-util)  
| 2: [Modding](#2-modding)  
|__ 2.0: [Custom Main](#20-custom-main)  
|__ 2.1: [Algorithms](#21-algorithms)  
|____ 2.1.0: [Simulation](#210-simulation)  
|____ 2.1.1: [Analysis](#211-analysis)  
|__ 2.2: [Model](#22-model)  
|__ 2.3: [I/O](#23-io)  
|____ 2.3.0: [Command Line Arguments](#230-command-line-arguments)  
|____ 2.3.1: [Input C++ and Header Files](#231-input-c-and-header-files)  
|____ 2.3.2: [Input CSV Files](#232-input-csv-files)  
|____ 2.3.3: [Output CSV Files](#233-output-csv-files)  
| 3: [Documentation](#3-documentation) ?  
|__ 3.n: for each folder  
|____ 3.n.n: for each file and class  
|______ 3.n.n.n: for each function and field  

## 0: Principle Paradigms

such as observer-observable, inheritance of simulation_base, etc

[Back to Top](#dde-documentation)

## 1: Organization

#### 1.0: /anlys

* "queen"
* only /sim can (should) access (by this I mean "#include") /anlys
* /anlys can access everyone except /sim

***
#### 1.1: /core

* "knights"
* everyone except /util can access /core
* /core can only access /util (after we get rid of model.*, the /io outlier will be eliminated)

***
#### 1.2: /io

* "nobles"
* only /anlys and /sim can access /io
* /io can only access /core and /util

***
#### 1.3: /sim

* "king"
* no one can access /sim
* /sim can access everyone

***
#### 1.4: /util

* "serfs"
* everyone can access /util
* /util cannot access anyone

[Back to Top](#dde-documentation)

## 2: Modding

#### 2.0: Custom Main

how to write your own main.cpp and add it to cmake

***
#### 2.1: Algorithms

***
#### 2.1.0: Simulation

how to add new simulation class + implementation of context

***
#### 2.1.1: Analysis

using observer interface and/or analysis inheritance to do analyses

***
#### 2.2: Model

see section 2 of 'README.md'. RECOMPILE IS NECESSARY!!! reiterate that each compile is model-specific

***
#### 2.3: I/O

***
#### 2.3.0: Command Line Arguments

how to add/remove. how they're designed. intended design pattern/usage

***
#### 2.3.1: Input C++ and Header Files

add these to models folder, mention xmacros and suggest resources for learning them

***
#### 2.3.2: Input CSV Files

csvr description

***
#### 2.3.3: Output CSV Files

csvw description

[Back to Top](#dde-documentation)

## 3: Documentation

do we need this section?

[Back to Top](#dde-documentation)
