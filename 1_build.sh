CCPer="mpicc"
CCFlag="-std=c99"
CCLAPACKLIB="/opt/homebrew/opt/lapack/lib"
CCBLASLIB="/opt/homebrew/opt/openblas/lib"
CCLAPACKINC="/opt/homebrew/opt/lapack/include"
CCBLASINC="/opt/homebrew/opt/openblas/include"

${CCPer} -c mpiutil.c ${CCFlag} -I${CCLAPACKINC} -I${CCBLASINC}
${CCPer} -c tdma.c ${CCFlag} -I${CCLAPACKINC} -I${CCBLASINC}
${CCPer} -c main.c ${CCFlag} -I${CCLAPACKINC} -I${CCBLASINC}
${CCPer} *.o ${CCFlag} -I${CCBLASINC} -L${CCBLASLIB} -I${CCLAPACKINC} -L${CCLAPACKLIB} -llapacke -llapack -lblas

ls