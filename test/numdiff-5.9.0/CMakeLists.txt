set(CMAKE_BUILD_TYPE Debug)
add_executable(
    numdiff
    analyze.c
    arith.c
    bitvector.c
    cmpbuf.c
    cmpfns.c 
    error.c 
    exitfail.c 
    flags.c 
    getopt.c 
    getopt1.c 
    inout.c
    io.c 
    linesplit.c 
    main.c 
    new.c 
    numutil.c 
    options.c  
    setmode.c 
    side.c 
    thrlist.c 
    util.c 
    xalloc-die.c
    xmalloc.c 
)
target_compile_options(numdiff PRIVATE ${CMAKE_C_FLAGS_DEBUG})
#target_link_options(numdiff PRIVATE ${CMAKE_C_FLAGS_DEBUG})
