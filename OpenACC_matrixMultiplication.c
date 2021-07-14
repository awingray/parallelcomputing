/* File: matrixMultiplication.c
 * Written by Sofie Lovdal 31.5.2018 based on LUP submission for Parallel
 * Computing 2016 by S.L and Nadia Hartsuiker
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define real float

#define ABS(a) ((a)<0 ? (-(a)) : (a))

/* forward references */
static void *safeMalloc(size_t n);
static real **allocMatrix(size_t height, size_t width);
static void freeMatrix(real **mat);
static void showMatrix (size_t n, real **A);
/********************/

#pragma acc routine seq
static double timer(void)
{
  struct timeval tm;
  gettimeofday (&tm, NULL);
  return tm.tv_sec + tm.tv_usec/1000000.0;
}
#pragma acc routine seq
void *safeMalloc(size_t n)
{
  void *ptr;
  ptr = malloc(n);
  if (ptr == NULL)
  {
    fprintf (stderr, "Error: malloc(%lu) failed\n", n);
    exit(-1);
  }
  return ptr;
}


real **allocMatrix(size_t height, size_t width)
{
  real **matrix;
  size_t row;

  matrix = safeMalloc(height * sizeof(real *));
  matrix[0] = safeMalloc(width*height*sizeof(real));
  
   
  for (row=1; row<height; ++row) 
    matrix[row] = matrix[row-1] + width;
   
  return matrix;
}


void freeMatrix(real **mat)
{  
  free(mat[0]);
  free(mat);
}

void showMatrix (size_t n, real **A)
{
  size_t i, j;
  for (i=0; i<n; ++i)
  {
    for (j=0; j<n; ++j)
    {
      printf ("%f ", A[i][j]);
    }
    printf ("\n");
  }
}



real **multiplyMatrix (size_t n, real **A, real **B) { 
 real ** C;
 C = allocMatrix(n, n);
 int i, j, index;
 real sum=0.0;
 
	#pragma acc data present(A[0:n][:n],B[0:n][:n]) create(C[:n][:n]) 
	{ 
	#pragma acc parallel loop gang 
	for (i=0; i<n; i++){
		#pragma acc loop worker 
		 for (j=0; j<n; j++){
			 sum=0.0; 
			 #pragma acc loop vector reduction(+:sum)
			 for(index = 0; index<n; index++) {
				sum += A[i][index]*B[index][j];
			 } 
			 C[i][j] = sum; 
		 }
	 }
	 }//parallel 

 return C;
}	

int main(int argc, char **argv)
{
  real **A, **B, **C;
  double clock;
  int N, i, j;
  
  system("date"); system("echo $USER");
  
  printf ("Enter matrix size n: \n");
  scanf ("%d", &N);
   
  A = allocMatrix(N, N);
  B = allocMatrix(N, N);
  
  /*Initialize matrices*/

  #pragma acc parallel loop collapse(2) 
  for (i=0; i<N; i++) {
	  for (j=0; j<N; j++) {
		  if(i==j) {
			  A[i][j]=-2;
			  B[i][j]=-2;
		  } else if (j==i-1 || j==i+1) {
			  A[i][j]=1;
			  B[i][j]=1;
		  } else {
			  A[i][j]=0;
			  B[i][j]=0;
		  }
	  }
  }
 #pragma acc enter data copyin(A[0:N*N],B[0:N*N])
 
  
  /* start timer: only measure the runtime of the core algorithm */
  clock = timer();
   C = multiplyMatrix(N, A, B);
  
  /* stop timer */
  clock = timer() - clock;
  printf ("wallclock: %lf seconds\n", clock);
 
  printf("C[0][0]= %f \n", C[0][0]);
  printf("C[0][1]= %f \n", C[0][1]);
  printf("C[0][2]= %f \n", C[0][2]);
  printf("C[0][3]= %f \n", C[0][3]);
  printf("C[1][1]= %f \n", C[1][1]);
  
 
  
  #pragma acc exit data delete(A[0:N*N],B[0:N*N])
  
  freeMatrix(A);
  freeMatrix(B);
  freeMatrix(C);
  
  return EXIT_SUCCESS;
}
