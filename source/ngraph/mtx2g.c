
/* This program converts sparse matrix files into edge-list graphs, i.e. it
converts Matrix Market (mtx) sparse matrices into '*.g' files (edge-lists)
*/


#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"


int main(int argc, char *argv[])
{
    int ret_code;
    MM_typecode matcode;
    int M, N, nz;   
    int i, I, J;
    double val, real_part, img_part;;


    if (mm_read_banner(stdin, &matcode) != 0)
    {
        exit(1);
    }

    if (!mm_is_sparse(matcode))
        exit(1);

      /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(stdin, &M, &N, &nz)) !=0)
        exit(1);

 

    if (mm_is_pattern(matcode))
    {
        for (i=0; i<nz; i++)
        {
            scanf("%d %d \n", &I, &J);
            if (I != J)
                printf("%d %d\n", I, J);
        }
            
    
    }
    else if (mm_is_real(matcode))
    {
        for (i=0; i<nz; i++)
        {
            scanf("%d %d %lg\n", &I, &J, &val);
            if (I != J)
                printf("%d %d\n", I, J);
        }
    }
    else if (mm_is_complex(matcode))
    {
        for (i=0; i<nz; i++)
        {
            scanf("%d %d %lg %lg\n", &I, &J, &real_part, &img_part);
            if (I != J)
                printf("%d %d\n", I, J);
        }
    }
   
    return 0;
}
