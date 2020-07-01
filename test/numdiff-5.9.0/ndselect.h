/*
    Numdiff - compare putatively similar files, 
    ignoring small numeric differences
    Copyright (C) 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017  Ivano Primi  <ivprimi@libero.it>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef _NDSELECT_H_
#define _NDSELECT_H_

#include"config.h"
#include"bitvector.h"

/* Error codes */
#define OPEN_ERROR     -1
#define READ_ERROR     -2
#define CLOSE_ERROR    -3

typedef struct {
  /* Mask of the options */
  bitvector optmask;

  /* Begin, end, step */
  unsigned long begin_line, end_line, step;

  /* First field, last field, increment */
  unsigned long first_field, last_field, increment;

  /* Internal fields separators */
  char **ifs;

  /* Output separator */
  char *osep;

  /* File to read from */
  const char *file;
} Argslist ; /* A structure of this type is used to store the options */
/* set by the user                                                     */ 

enum {
  ___H_MASK = 0, /* -h option, used to recall help */
  ___B_MASK = 1, /* -b option, used to set the start line */
  ___E_MASK = 2, /* -e option, used to set the end line */
  ___S_MASK = 3, /* -s option, used to explicitly set the step */
  ___SF_MASK= 4, /* -F option, used to set the first field to display */
  ___SL_MASK= 5, /* -L option, used to set the last field to display */
  ___SI_MASK= 6, /* -I option, used to explicitly set the increment */
  ___SS_MASK= 7, /* -S option, used to explicitly set IFS */
  ___SD_MASK= 8, /* -D option, used to explicitly set IFS */
  ___SO_MASK= 9, /* -O option, used to set the output separator */
  ___X_MASK =10, /* -x option, to omit empty lines */
  ___L_MASK =11, /* -l option, to redirect the standard error on a file */
  ___O_MASK =12, /* -o option, to redirect the standard output on a file */
  ___V_MASK =13, /* -v option, used to show version number,
		    Copyright and No-Warranty */
  MAX_NDSELECT_OPTIONS = 20  
};
/* I18N and L10N support */

#ifdef ENABLE_NLS
#include <libintl.h>
#define _(String) gettext (String)
#define gettext_noop(String) String
#define N_(String) gettext_noop (String)
#else
#define _(String) (String)
#define N_(String) String
#define textdomain(Domain)
#define bindtextdomain(Package, Directory)
#endif

#ifndef PACKAGE2
#define PACKAGE2 "ndselect"
#endif

#ifndef LOCALEDIR
#define LOCALEDIR "/usr/local/share/locale/"
#endif

#endif /* _NDSELECT_H_ */
