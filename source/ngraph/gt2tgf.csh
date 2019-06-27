#!/bin/csh
#
# convert .g and .t file into a TGF (.tgf) file for use with yEd
#
# usage: gt2mt foo.g  foo.t  > foo.mt
#
  set gfile = $argv[1]
  set tfile = $argv[2]
 
  cat $tfile
  echo '#'
  cat $gfile 

