#include<mpi.h> 

int mpiutil_para(int,int,int,int,int*,int*);
void mpiutil_pack_d(double *,double *, int[][3],int[][3],int[3], MPI_Comm);
void mpiutil_unpack_d(double *,double *, int[][3],int[][3],int[3], MPI_Comm);
void mpiutil_pack_info(int ,int ,int ,int [][3],int [][3], MPI_Comm);
void mpiutil_unpack_info(int ,int ,int ,int [][3],int [][3], MPI_Comm);
void mpiutil_filewrite(int,int,int,double*,double*,double*,double*,int,int);
void mpiutil_alltoallv(double *,int[],int[], double *,int[],int[], MPI_Comm); //미완성
