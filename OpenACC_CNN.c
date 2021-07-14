/* File: CNN.c
 * Sofie Lovdal 31.5.2018
 * Implements the first convolutional layer of a primitive CNN similar to AlexNet
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <openacc.h> 

#define DEPTH 96
#define WIDTH 224
#define HEIGHT 224
#define WIDTHKERNEL 11
#define HEIGHTKERNEL 11

#pragma acc routine seq 
static double timer(void)
{
  struct timeval tm;
  gettimeofday (&tm, NULL);
  return tm.tv_sec + tm.tv_usec/1000000.0;
}


void *safeMalloc(size_t n)
{
  void *ptr;
  ptr = malloc(n);
  if (ptr == NULL)
  {
    //fprintf (stderr, "Error: malloc(%lu) failed\n", n);
    exit(-1);
  }
  return ptr;
}


int **allocMatrix(size_t height, size_t width)
{
  int **matrix;
  size_t row;

  matrix = safeMalloc(height * sizeof(int *));
  matrix[0] = safeMalloc(width*height*sizeof(int));
  for (row=1; row<height; ++row) 
    matrix[row] = matrix[row-1] + width;\
   
  #pragma acc enter data create(matrix[0:height][0:width*height])
  return matrix;
}


int ***alloc3DMatrix(size_t depth, size_t width, size_t height) {
  int k, i;
  int ***matrix = malloc(depth*sizeof(int **));
  for (k=0; k<depth; k++) {
    matrix[k] = malloc(height*sizeof(int *));
    matrix[k][0] = malloc(height*width*sizeof(int *));
    for (i=1; i<height; i++) {
      matrix[k][i] = matrix[k][i-1] + width;
    }
  }
  
  #pragma acc enter data create(matrix[:depth][:height][:width])
  return matrix;
}

void showMatrix (size_t n, int **A)
{
  size_t i, j;
  for (i=0; i<n; ++i)
  {
    for (j=0; j<n; ++j)
    {
      printf ("%d ", A[i][j]);
    }
    printf ("\n");
  }
}	


void freeMatrix(int **mat)
{
  #pragma acc exit data delete(mat[0:HEIGHT])
  #pragma acc exit data delete(mat) 
  free(mat[0]);
  free(mat);
}

void free3DMatrix(int ***mat, int depth) {
    int i;
    #pragma acc exit data delete(mat[0:DEPTH])
    #pragma acc exit data delete(mat) 
    for(i=0;i<depth;i++)
    {
        free(mat[i][0]);
        free(mat[i]);
    }
    free(mat);
}	

#pragma acc routine seq 
int max(int a, int b, int c, int d) {
	int max1, max2;
	max1=a>b? a: b;
	max2=c>d? c: d;
	return (max1>max2? max1 : max2);
}	

int ***applyMaxPooling (int ***C1) {
	/*Insert code here. This routine should return a 3D matrix D*/
	int i,j, k, l, maxTemp=0; 
	int poolDim =4;
	int ***temp = alloc3DMatrix(DEPTH,HEIGHT,WIDTH); 
	int max4[4]; 
	int dx=0, dy=0; 
	

	#pragma acc parallel present(C1[:DEPTH][:HEIGHT][:DEPTH]) 
	{
	#pragma acc loop gang independent
	for (i=0;i<DEPTH; i++) {//depth
		dx=0; 
		#pragma acc loop worker 
		for(j = 5; j< HEIGHT-poolDim-5;j+=poolDim) {//height
		    dy=0; 
		    #pragma acc loop vector collapse(2) reduction(max:maxTemp)
			for (k =5; k<WIDTH-poolDim-5; k+=poolDim) {//width
					for(l=0; l<poolDim; l++){
							maxTemp = max(C1[i][j+l][k+0],
								C1[i][j+l][k+1],
								C1[i][j+l][k+2],
								C1[i][j+l][k+3]); 
						    
						//max4[l] = maxTemp;
						
						}
						//max(max4[0],max4[1],max4[2],max4[3]);
						temp[i][dx][dy] = maxTemp; 
						dy++; 
					}
				  dx++;  
			}
			
	 }
 }//parallel 
	
	return temp; 
} 	


void convolve(int** A, int ***W1, int ***C1) {
	int i, j, k, l, m, activation=0;
	

    #pragma acc parallel present(A[:HEIGHT][:WIDTH], W1[:DEPTH][:WIDTHKERNEL][:HEIGHTKERNEL], C1[:DEPTH][:HEIGHT][:WIDTH])
    { 
    #pragma acc loop gang 
	for (k = 0; k < DEPTH; k++) {
		#pragma acc loop collapse(2) worker independent 
		for (i = 5; i < HEIGHT-5; i++) {    
			for (j = 5; j < WIDTH-5; j++) {
				activation =0;
				#pragma acc loop vector reduction(+:activation) independent 
				for (l = -5; l < 6; l++) {
					   for (m = -5; m < 6; m++) {
						  activation += W1[k][l+5][m+5]*A[i+l][j+m];
					   }
				}
					
				if (activation < 0){
				   C1[k][i][j] = 0;
				}else{ 
					C1[k][i][j] = activation; 
				}  /*ReLu computation */
			}
		 }
	 }
     }//for parallel 

	
  }


void initData(int **A) {
	int i, j;
	
	for(i=0; i<HEIGHT; i++) {
		for(j=0; j<WIDTH; j++) {
		   if (j==i-1 || j==i+1 || j==i || j==i-2 || j==i+2) {
		      A[i][j]=0;
		   } else {
			  A[i][j]=i*j;   
		   }		
		}
	}	
	#pragma acc update device(A[0:HEIGHT][0:WIDTH])		
}

void initConvolutionKernels(int ***W1) {
	int i, j, k;
		
	
	for(k=0; k<DEPTH; k++) {
		for(i=0; i<WIDTHKERNEL; i++) {
			for(j=0; j<HEIGHTKERNEL; j++) {
				if(i==5 && j==5) {
					W1[k][i][j]=(i+1)*k;
				} else if(i>=4 && i<=6 && j>=4 && j<=6) {
					W1[k][i][j] = -1*(k+i);
				} else if(i==5 || j==5) {
					W1[k][i][j] = -1;
				} else {
					W1[k][i][j] = 0;
				}
			}
		}

	}
	
	#pragma acc update device(W1[:DEPTH][:WIDTHKERNEL][:HEIGHTKERNEL])
						   
} 

int main(int argc, char **argv)
{
  int **A;
  int ***W1, ***C1, ***D;
  double clock;
  
  system("date"); system("echo $USER");
  
  A=allocMatrix(WIDTH, HEIGHT);
  W1=alloc3DMatrix(DEPTH, WIDTHKERNEL, HEIGHTKERNEL);
  C1 = alloc3DMatrix(DEPTH, WIDTH, HEIGHT);

  initData(A);
  initConvolutionKernels(W1);
  
  /* start timer: only measure the runtime of the core algorithm */
  clock = timer();
  convolve(A, W1, C1);
 
  /* stop timer */
  clock = timer() - clock;
  printf ("wallclock: %lf seconds\n", clock);
  D = applyMaxPooling(C1); 
  printf("show\n");
  showMatrix(55, D[54]);

  
  freeMatrix(A);
  free3DMatrix(W1, DEPTH);
  free3DMatrix(C1, DEPTH);
  free3DMatrix(D,DEPTH); 
  return EXIT_SUCCESS;
}

