// File: contrast.c
// Written by Arnold Meijster and Rob de Bruin.
// Restructured by Yannick Stoffers.
// 
// A simple program for contrast stretching PPM images. 

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "image.h"

// Stretches the contrast of an image. 


static void *safeMalloc(int n)
{ /* wrapper function for malloc with error checking */
	void *ptr = malloc(n);
	if (ptr == NULL)
	{
		printf("Error: memory allocation failed.\n");
	    abort();
	}
	return ptr;
}

void minMax(int *data, int length, int *min, int *max){ 
	
	 
	*min = 255; 
	*max = 0;
	
	// Determine minimum and maximum.

	for (int i=0; i<length; ++i)
	{
		*min = (data[i] < *min ? data[i] : *min);
		*max = (data[i] > *max ? data[i] : *max);
	}
	
}

void contrastStretchByData(int *data, int length, float scale){
	 
	
	for (int i=0; i<length; ++i)
		data[i] = scale * (data[i]);
	}
	
	

int main (int argc, char **argv)
{
	Image image;
	int rank, size, i ,*sendcnt,scale ,*data, *offset, portion;
	int min, max, fMin, fMax; 
    
	// Initialise MPI environment.
	MPI_Init (&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Status stat; 
	double time = MPI_Wtime();
	
	sendcnt = safeMalloc(sizeof(int)*size);
	offset = safeMalloc(sizeof(int)*size) ; 
	
	
	
	if(rank ==0 ){ //master  
		 // Prints current date and user. DO NOT MODIFY
		 system("date"); system("echo $USER");
		image = readImage (argv[1]);
		portion = (image->height*image->width)/size; 
		data = safeMalloc(sizeof(int)*portion);
		if (argc != 3)
		{
			printf ("Usage: %s input.pgm output.pgm\n", argv[0]);
			exit (EXIT_FAILURE);
		}
		for (i = 0;  i < size ; i++) {
			//last one might not have equal size 
			if(rank == (size-1)) {
			  sendcnt[i] = (image->height*image->width) -(size-1)* portion; 
			}else{ 
			  offset[i] = i*portion; 
			  sendcnt[i] = portion; 
			}
			if(i > 0 ){  
				MPI_Send(&sendcnt[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			}
		
		} 
		
		printf("distribute to slaves\n");
		//Comm time for scattering 
		double Stime = MPI_Wtime();
		MPI_Scatterv(image->imdata[0],sendcnt,offset, MPI_INT, data,portion, MPI_INT, 0, MPI_COMM_WORLD);
		printf( "Time for MPI_Scatterv %d : %f\n", rank, MPI_Wtime()-Stime);
		
		minMax(data,portion,&min, &max);
		
		MPI_Allreduce(&min, &fMin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
	    MPI_Allreduce(&max, &fMax, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	  
	    printf("contrastStretch\n");
	 
	    contrastStretchByData(data, portion, (float)(255-0)/(fMax-fMin)); 
	    double Gtime = MPI_Wtime();
	    MPI_Gatherv(data, portion, MPI_INT, image->imdata[0], sendcnt, offset, MPI_INT, 0, MPI_COMM_WORLD); 
	    printf( "Time for MPI_Gatherv %d : %f\n", rank, MPI_Wtime()-Gtime);
	     printf("writing\n"); 
	     writeImage(image, argv[2]);
	     freeImage (image);
	    free(data);
	   free(offset); 
	   free(sendcnt); 
	}else{//slave  
		MPI_Recv(&portion, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &stat);
		data = safeMalloc(sizeof(int)*portion);
		
		MPI_Scatterv(NULL, NULL, NULL, MPI_INT, data, portion, MPI_INT, 0,MPI_COMM_WORLD);
		minMax(data, portion, &min, &max);
		MPI_Allreduce(&min, &fMin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
	    MPI_Allreduce(&max, &fMax, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	    
	    
	    contrastStretchByData(data, portion,(float)(255-0)/(fMax-fMin));
	    MPI_Gatherv(data, portion, MPI_INT, NULL, NULL, NULL, MPI_INT, 0,MPI_COMM_WORLD);
	    
	}
	
	  
	printf( "Time for %d : %f\n", rank, MPI_Wtime()-time); 

	
	// Finalise MPI environment.
	MPI_Finalize ();
	return EXIT_SUCCESS;
}
