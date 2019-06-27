#!/bin/csh
#
# convert 0-based .g and .t files into an URL-annotated .mt file
#
# usage: gt2mt foo.g  foo.t  > foo.mt
#
  set gfile = $argv[1]
  set tfile = $argv[2]
  cat $gfile | community_fast.py | sort -n | mi2mt $tfile


