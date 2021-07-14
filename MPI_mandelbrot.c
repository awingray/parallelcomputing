// File: mandelbrot.c
// Written by Arnold Meijster and Rob de Bruin
// Restructured by Yannick Stoffers
// 
// A simple program for computing images of the Mandelbrot set. 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "fractalimage.h"

#define WIDTH  4096
#define HEIGHT 3072
#define MAXITER 3000

int rank, size;

// Function for computing mandelbrot fractals.
void mandelbrotSet (double centerX, double centerY, double scale, int portion, Image image)
{
	int w = image->width, h = image->height, **im = image->imdata;
	double a, b;
	double x, y, z;
	int i, j, k;
	
	for (i = rank*portion; i < (rank+1)*portion; i++)
	{
		b = centerY + i * scale - ((h / 2) * scale);
		for (j = 0; j < w; j++)
		{
			a = centerX + j * scale - ((w / 2) * scale);
			x = a;
			y = b;
			k = 0;
			while ((x * x + y * y <= 100) && (k < MAXITER))
			{
				z = x;
				x = x * x - y * y + a;
				y = 2 * z * y + b;
				k++;
			}
			im[i][j] = k;
		}
	}
}

int main (int argc, char **argv)
{
	Image mandelbrot;
	MPI_Status stat; 
	
	// Prints current date and user. DO NOT MODIFY
	
	
	// Initialise MPI environment.
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	MPI_Comm_size (MPI_COMM_WORLD, &size);
    if(rank ==0) system("date"); system("echo $USER");
    mandelbrot = makeImage (WIDTH, HEIGHT);
    
    int portion = mandelbrot->height /size; 
    int start = rank * portion; 
    double time  = MPI_Wtime(); 
    
	mandelbrotSet (-0.65, 0, 2.5 / HEIGHT, portion, mandelbrot);
	printf("End time by %d: %f\n", rank, MPI_Wtime()-time);
	
	
	if (rank==0) { //master
		for (int i = start+portion;  i <mandelbrot->height ; i++) { 
			MPI_Recv(mandelbrot->imdata[i], mandelbrot->width, MPI_INT, MPI_ANY_SOURCE, i, MPI_COMM_WORLD, &stat) ;
		}
		writeImage (mandelbrot, "mandelbrot.ppm", MAXITER);
	}else{ //slave 
		for (int i =start; i<start+portion; i++){ 
			MPI_Send(mandelbrot->imdata[i], mandelbrot->width, MPI_INT, 0, i, MPI_COMM_WORLD);
		}
	}
		
		
	
	

	freeImage (mandelbrot);
	
	// Finalise MPI environment.
	MPI_Finalize ();
	return EXIT_SUCCESS;
}
