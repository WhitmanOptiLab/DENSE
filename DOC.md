# DDE Documentation
Notes on the design principles, organization, and modability of the DDE simulation.

Contributors to this file should be aware of [Adam Pritchard's Markdown Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet).

## Table of Contents

| 0: [Principle Paradigms](#0-principle-paradigms)  
| 1: [Organization](#1-organization)  
|__ 1.0: [/anlys](#10-anlys)  
|__ 1.1: [/core](#11-model)  
|__ 1.2: [/io](#12-io)  
|__ 1.3: [/sim](#13-sim)  
|__ 1.4: [/util](#14-util)  
|__ 1.5: [Feudal Metaphor](#15-feudal-metaphor)  
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

High-level overview of anlys  
* only /sim can (should) access (by this I mean "#include") /anlys
* /anlys can access everyone except /sim

***
#### 1.1: /core

High-level overview of core  
* everyone except /util can access /core
* /core can only access /util (after we get rid of model.*, the /io outlier will be eliminated)

***
#### 1.2: /io

High-level overview of io  
* only /anlys and /sim can access /io
* /io can only access /core and /util

***
#### 1.3: /sim

High-level overview of sim  
* no one can access /sim
* /sim can access everyone

***
#### 1.4: /util

High-level overview of util  
* everyone can access /util
* /util cannot access anyone

***
#### 1.5: Feudal Metaphor

Hopefully this metaphor isn't futile.

`/sim` is the king. An essential component of the kingdom, especially if there is no queen to rule the realm.  
`/anlys` is the queen. She is the cunning side-kick of the king, but she can also run the show if the king is absent.  
`/io` is the vassal. He is the eyes and ears of the royal family. Without him, the kingdom *can* operate, though not to its fullest potential.  
`/core` is the knight. He conducts the bulk of the kingdom's work.  
`/util` is the peasant. He and his family provides the basic resources for the kingdom.

Each member of this society can access to everyone lower than them, but cannot access anyone higher than them. Breaking these societal rules will not result in death, but may result in you getting shunned by your fellow countrymen.

[Back to Top](#dde-documentation)

## 2: Modding

#### 2.0: Custom Main

To circumnavigate the argument parsing structure of main.cpp and create a custom main, create a new file with a function `int main(int argc, char* argv[])` insize the `/source` directory. To take advantage of `arg_parse`, be sure to have `arg_parse::init(argc, argv);` at or near the beginning of `main`.

In order for CMake to generate an executable using the new `main`, various lines must be added to `/source/CMakeLists.txt`. Copy and paste all the lines between and including `function(SIMULATION localname simdir)` and `endfunction(SIMULATION localname simdir)`. In your new CMake function, replace the two occurances of `SIMULATION` with your own name. Next, create a CMake function call after this line, `SIMULATION(simulation ${PROJECT_BINARY_DIR})`. Copy that line, paste it directly below, then replace `SIMULATION` with the name you gave your function earlier and replace `simulation` with what you want the resulting executable to be named. Optionally, to guarantee that the `*_template.csv` files get generated upon building the new `main`, copy the line, `add_dependencies(simulation csv_gen_run)`, then paste it directly below. Replace `simulation` with the executable name you gave earlier. Now, whenever `make` is run, the new executable will also be built.

***
#### 2.1: Algorithms

***
#### 2.1.0: Simulation

To add an entirely new simulation class, the user must make sure that the class inherits from `simulation_base` and calls the base constructor in its own constructor.  The simulation class must also define its own context in the header file inside that class that inherits from `ContextBase`.

To notify observers (analysis and file-writing objects) throughout simulation, the class must regularly call `notify(ContextBase& context_name)` in the function `simulate()`, passing it an instanciation of the context defined in the header file. If the class ever calls `notify(ContextBase& context_name)`, then it MUST, before `simulate()` concludes call the function `finalize()`.

***
#### 2.1.1: Analysis

Any object that performs a regular evaluation of concentration levels throughout simulation is considered an analysis. To create a new analysis, make a class in `source/anlys` that inherits from `Analysis` (include `base.hpp`) and calls the base constructor in its own constructor.  Because `Analysis` inherits from `Observer`, it must define the functions `update(ContextBase& context_name)` and `finalize()`.

***
#### 2.2: Model

To create a new model make a copy of the directory `template_model`. In all four files, follow directions in `README.md` section 2 to implement the new model. __*THE SYSTEM MUST BE RECOMPILED*__ with `cmake ..` and then `make`, because each compile is model-specific.

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
