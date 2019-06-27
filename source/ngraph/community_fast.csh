#!/bin/csh
#
# convert arbitrary .g and .t files into an URL-annotated .mt file

foreach file ($argv)
  echo processing $file
  set t_file = $file:r:r.t
  set mt_output_file = $file:r:r.mt

  # temporary files
  #
  #set g0_output_file = $file:r:r.g0
  #set imap_output_file = $file:r:r.imap
  
  set g0_output_file = /tmp/tempg0.$$
  set imap_output_file = /tmp/tempimap.$$

  cat $file | g2g0 $g0_output_file $imap_output_file
  date
  cat $g0_output_file | community_fast.py | mi2mt -m $imap_output_file $t_file | sort -n  > $mt_output_file
  date
  rm $g0_output_file
  rm $imap_output_file
end


