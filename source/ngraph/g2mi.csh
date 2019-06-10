#!/bin/csh
#
# convert arbitrary .g files into a community index
#
# usage: cat foo.g  | g2mi  > foo.mi


 # temporary files: 0-based g, and mapping to 0-based indices
 #
 set g0_tmp = /tmp/tempg0.$$
 set imap0_tmp = /tmp/tempimap0.$$


 g2g0 $g0_tmp $imap0_tmp
 cat $g0_tmp | community_fast.py | mi0_2mi $imap0_tmp | sort -n 

 rm $g0_tmp
 rm $imap0_tmp

