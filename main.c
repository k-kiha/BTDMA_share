/*              many tdma = nsys
                ----------->                                
    |    11   12   13   14   15   16   17   |
    |    21   22   23   24   25   26   27   | 
nsub|    31   32   33   34   35   36   37   | 
    |    41   42   43   44   45   46   47   | 
    |    51   52   53   54   55   56   57   | #grid
   ---  ----------------------------------  | = n
    |    61   62   63   64   65   66   67   | 
    |    71   72   73   74   75   76   77   |
nsub|    81   82   83   84   85   86   87   |
    |    91   92   93   94   95   96   97   |
    |   101  102  103  104  105  106  107   |
   ---  ----------------------------------  | 
    |   111  112  113  114  115  116  117   |  solving direction
nsub|   121  122  123  124  125  126  127   V 
    |   131  132  133  134  135  136  137    
    |   141  142  143  144  145  146  147    
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include<mpi.h> 
#include "tdma.h"
#include "mpiutil.h"


/* ------ For demo ------*/
int init_inputs(int, int, int, double *, double *, double *, double *, int, int);
int check_result(int, int, int, double *, int);
/* ----------------------*/

int main(){
    int myrank, nprocs;
    int n=14,m=5,nsys=7;
    int nsub;
    double *a,*b,*c,*d;

    int indxtmp_a, indxtmp_b;

    double wtimea,wtimeb;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    nsub = mpiutil_para(0,n-1,myrank,nprocs, &indxtmp_a, &indxtmp_b);

    /*  block matrices(m by m): "Column-Major order!!"
        memory: last ----> first
              a[nsub][nsys][m][m]   :off diagonal(lower)
              b[nsub][nsys][m][m]   :diagonal
              c[nsub][nsys][m][m]   :off diagonal(upper)
              d[nsub][nsys][m]      :RHS(in) and Solution(out)
    */
    a = (double *)malloc(sizeof(double)*nsub*nsys*(m*m)); 
    b = (double *)malloc(sizeof(double)*nsub*nsys*(m*m)); 
    c = (double *)malloc(sizeof(double)*nsub*nsys*(m*m)); 
    d = (double *)malloc(sizeof(double)*nsub*nsys*(m)); 

    init_inputs(nsub, nsys, m, a, b, c, d, myrank, nprocs);
    
    wtimea = MPI_Wtime();
    
    if(nprocs==1){
        btdma_many(n, nsys, m, a,b,c,d);
    }else{
        BTDMA_PLAN tttplan;
        btdma_makeplan(nsub,nsys,m, &tttplan, MPI_COMM_WORLD);
        btdma_many_mpi(nsub, nsys, m, a, b, c, d, &tttplan);
        btdma_cleanplan(&tttplan);
    }
    
    wtimeb = MPI_Wtime();
    printf("wtime=%20.15f @%03d/%2drank; n=%3d nsys=%3d m=%3d\n",wtimeb-wtimea,myrank,nprocs,n,nsys,m);
    check_result(nsub, nsys, m, d, myrank);

    free(a);
    free(b);
    free(c);
    free(d);

    MPI_Finalize();
    return 0;
}


/* ------ For demo ------*/
    int init_inputs(int n, int nsys, int m, double *a, double *b, double *c, double *d, int myrank, int nprocs){
        int ij;
        //m by m block matrix: eigen values= 1.e-8, 1.e+0, 1.e+1, 1.e+2, 1.e+3 
        double testM[5][5]={ 150643./937.,  49745./469.,  21470./967., -19012./547., 271490./861.,
                              49745./469.,  38514./529.,   8501./430., -20059./813., 193592./923.,
                              21470./967.,   8501./430.,  13343./283., -20363./765.,  -3425./148.,
                             -19012./547., -20059./813., -20363./765.,  11909./619., -17911./688.,
                             271490./861., 193592./923.,  -3425./148., -17911./688., 706412./871.};

        double testV[5]={16285592618779688. / 28591131662331.,
                         30717996443230039. / 80055139188570.,
                            10488105146315. /   266461526412.,
                         -4483767319405157. / 48294440562960.,
                          1744303204807789. /  1355407291056.};

        //solution == {1.,1.,1.,...,1.}
        for(int q = 0; q < n; q++){
            for(int sys = 0; sys < nsys; sys++){
                // LAPACK_COL_MAJOR
                for(int i = 0; i < m; i++){        
                    d[i + sys*(m)+ q*(m*nsys)] = -testV[i] ;
                    if(q==0  && myrank ==0       ) d[i + sys*(m)+ q*(m*nsys)] = -2*testV[i] ;
                    if(q==n-1&& myrank ==nprocs-1) d[i + sys*(m)+ q*(m*nsys)] = -2*testV[i] ;
                }

                for(int j = 0; j < m; j++){
                    for(int i = 0; i < m; i++){
                        // LAPACK_COL_MAJOR
                        ij = i + m*j;
                        a[ij + sys*(m*m)+ q*(m*m*nsys)] = 1.*testM[i][j];
                        b[ij + sys*(m*m)+ q*(m*m*nsys)] =-3.*testM[i][j];
                        c[ij + sys*(m*m)+ q*(m*m*nsys)] = 1.*testM[i][j];
                    }
                }
            }
        }

        return 0;
    }

    int check_result(int n, int nsys, int m, double *d, int myrank){
        int ij;
        char filename[20];
        FILE *file;
        sprintf(filename, "BBB%03d.csv", myrank);

        file = fopen(filename, "w");  
        for(int q = 0; q < n; q++){
            for(int i = 0; i < m; i++){
                fprintf(file,"%3d::%2d=",q,i);
                for(int sys = 0; sys < nsys; sys++){    
                    fprintf(file,"%26.16e,",d[i + sys*(m)+ q*(m*nsys)]);
                }
                fprintf(file,"\n");
            }
        }

        fclose(file);
        return 0;
    }

/* ----------------------*/
