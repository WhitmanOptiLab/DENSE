
/* This program converts graph (edge list) files into a directed GraphViz
   .dot file in the form

digraph G {
1->6 ;
1->11 ;
2->6 ;
2->9 ;
2->11 ;
3->4 ;
}

*/


#include <stdio.h>


int main(int argc, char *argv[])
{
    int I, J;



    /* write the header for GraphViz files */
    printf("digraph G {\n");

    while (scanf("%d %d \n", &I, &J) != EOF)
            printf("%d -> %d ;\n", I, J);

    printf("}\n");

   
    return 0;
}
