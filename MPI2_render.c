// File: render.c
// Written by Arnold Meijster and Rob de Bruin.
// Restructured by Yannick Stoffers.
// 
// A simple orthogonal maximum intensity projection volume render. 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "image.h"
#include "volume.h"
#include <mpi.h>

#define NFRAMES 360

void rotateVolume (double rotx, double roty, double rotz, Volume volume, Volume rotvolume)
{
	// Rotate the volume around the x-axis with angle rotx, followed by a 
	// rotation around the y-axis with angle roty, and finally around the 
	// z-axis with angle rotz. The rotated volume is returned in rotvolume.
	int i, j, k, xi, yi, zi;
	int width  = volume->width;
	int height = volume->height;
	int depth  = volume->depth;
	byte ***vol = volume->voldata;
	byte ***rot = rotvolume->voldata;
	double x, y, z;
	double sinx, siny, sinz;
	double cosx, cosy, cosz;
	
	for (i = 0; i < depth; i++)
		for (j = 0; j < height; j++)
			for (k = 0; k < width; k++)
				rot[i][j][k] = 0;

	sinx = sin (rotx);  siny = sin (roty); sinz = sin (rotz);
	cosx = cos (rotx);  cosy = cos (roty); cosz = cos (rotz);
	for (i = 0; i < depth; i++)
	{
		for (j = 0; j < height; j++)
		{
			for (k = 0; k < width; k++)
			{
				xi = j - height / 2;
				yi = k - width / 2;
				zi = i - depth / 2;

				// Rotation around x-axis.
				x = (double)xi;
				y = (double)(yi * cosx + zi * sinx);
				z = (double)(zi * cosx - yi * sinx);
				xi = (int)x;
				yi = (int)y;
				zi = (int)z;
				// Rotation around y-axis.
				x = (double)(xi * cosy + zi * siny);
				y = (double)(yi);
				z = (double)(zi * cosy - xi * siny);
				xi = (int)x;
				yi = (int)y;
				zi = (int)z;

				// Rotation around z-axis.
				x = (double)(xi * cosz + yi * sinz);
				y = (double)(yi * cosz - xi * sinz);
				z = (double)zi;

				xi = (int)(x + height / 2);
				yi = (int)(y + width / 2);
				zi = (int)(z + depth / 2);
				if ((xi >= 0) && (xi < height) && (yi >= 0) && (yi < width) && (zi >= 0) && (zi < depth))
					rot[zi][xi][yi] = vol[i][j][k];
			}
		}
	}
}

void contrastStretch (int low, int high, Image image)
{
	// Stretch the dynamic range of the image to the range [low..high].
	int row, col, min, max;
	int width = image->width, height = image->height, **im = image->imdata;
	double scale;
		
	// Determine minimum and maximum.
	min = max = im[0][0];
	for (row = 0; row < height; row++)
	{
		for (col = 0; col < width; col++)
		{
			min = im[row][col] < min ? im[row][col] : min;
			max = im[row][col] > max ? im[row][col] : max;      
		}
	}
	
	// Compute scale factor.
	scale = (double)(high - low) / (max - min);

	// Stretch image.
	for (row = 0; row < height; row++)
		for (col = 0; col < width; col++)
			im[row][col] = (int)(scale * (im[row][col] - min));
}

void orthoGraphicRenderer (Volume volume, Image image)
{
	// Render image from volume (othographic maximum intensity projection).
	int i, j, k;
	int width = volume->width;
	int height = volume->height;
	int depth = volume->depth;                           
	int **im = image->imdata;
	byte ***vol = volume->voldata;

	for (i = 0; i < height; i++)
		for (j = 0; j < width; j++)
			im[i][j] = 0;
	
	for (i=0; i<depth; i++)
		for (j=0; j<height; j++)
			for (k=0; k<width; k++)
				im[j][k] += vol[i][j][k];

	contrastStretch (0, 255, image);
}

void smoothImage (Image image, Image smooth)
{
	int width = image->width, height = image->height;
	int **im = image->imdata, **sm = smooth->imdata;
	int i, j, ii, jj, sum, cnt;

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			cnt = 0;
			sum = 0;
			for (ii = i-1; ii <= i+1; ii++)
			{
				if ((ii >= 0) && (ii < height))
				{
					for (jj = j-1; jj <= j+1; jj++)
					{
						if ((jj >= 0) && (jj < width) && (im[ii][jj] != 0))
						{
							cnt++;
							sum += im[ii][jj];
						}
					}
				}
			}
			sm[i][j] = cnt == 0 ? 0 : sum / cnt;
		}
	}
}

void computeFrame(int frame, double rotx, double roty, double rotz, 
    Volume vol, Volume rot, Image image, Image smooth)
{

	char fnm[256];
	rotateVolume (rotx, roty, rotz, vol, rot);
	
	orthoGraphicRenderer (rot, image);

	smoothImage (image, smooth);

	sprintf (fnm, "frame%04d.pgm", frame);
	writeImage (smooth, fnm);   


}

int main (int argc, char **argv)
{ 
	
    int size, rank, width, height, depth, i, id, busy, frame;

	MPI_Init(&argc, &argv); 
    MPI_Comm_size(MPI_COMM_WORLD,&size); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
	double time = MPI_Wtime(),pTime, extime=0, extimep,  rotx =0 , roty=0, rotz=0; 
	int TASK_TAG =1, RESULT_TAG =2; 
	MPI_Status stat;
    MPI_Request *requests;
    Volume vol, rot;  
    Image im, smooth;
    int *tasks, info[3];
    
    tasks    = malloc(size * sizeof(int));
    requests = malloc(size * sizeof(MPI_Request)); 
    
    
    
    if (argc != 2){
		fprintf (stderr, "Usage: %s <volume.vox>\n", argv[0]);
		exit (EXIT_FAILURE);
	}
	
	
	if(rank == 0) { //master 
		//read input 
		vol = readVolume(argv[1]);
		width  = info[0] = vol->width;
		height = info[1] = vol->height;
		depth  = info[2] = vol->depth;
		im = makeImage(info[0], info[1]);
    }
   
    MPI_Bcast(info, 3, MPI_INT, 0, MPI_COMM_WORLD);
	//allocate 
	width = info[0]; 
	height =info[1]; 
	depth = info[2]; 
	
	if(rank!=0) vol = makeVolume(width, height, depth);
    rot = makeVolume(width, height, depth);
    im  = makeImage(width, height);
    smooth = makeImage(width, height);
    
    MPI_Bcast(vol->voldata[0][0], width*height*depth, MPI_BYTE, 0, MPI_COMM_WORLD);

	if(rank ==0){ //master 
		
		//distribute first work 
		
		frame = 0;
		
		for(i= 1; i<size ; i++){
			MPI_Send(&frame, 1, MPI_INT, i, TASK_TAG, MPI_COMM_WORLD);
			//tasks[i] = frame++;
			frame++; 
			
		}
		
		for (i=0; i<size; ++i){
			MPI_Irecv(im->imdata[0], width*height, MPI_INT, i, RESULT_TAG,
					MPI_COMM_WORLD, &requests[i]);
		}
		
		busy = size - 1;
		extime = MPI_Wtime(); 
		while(busy > 0)
		{
			MPI_Waitany(size, requests, &id, &stat);
			// send next frame 
			if (frame < NFRAMES) // check if frame is in bound 
			{
				MPI_Send(&frame, 1, MPI_INT, id, TASK_TAG, MPI_COMM_WORLD);
				
				frame++; 
				MPI_Irecv(im->imdata[0], width*height, MPI_INT, id,
						RESULT_TAG, MPI_COMM_WORLD, &requests[id]);
			}
			// stopping signal 
			else
			{
				i = -1;
				MPI_Send(&i, 1, MPI_INT, id, TASK_TAG, MPI_COMM_WORLD);
				busy--;
			}
       
		}
		extime = MPI_Wtime()-extime;

		free(tasks);
		free(requests);
		
	}else{ //slave
	
		while (1) //infinite loop
		{
			MPI_Recv(&frame, 1, MPI_INT, 0, TASK_TAG, MPI_COMM_WORLD, &stat);
			// stopping singal recieved 
			if (frame ==-1 )
				break;
			// compute frame 
			
			rotx = roty =rotz = 0; 
			switch (3*frame/NFRAMES)
			{
				case 0: rotx = 6*M_PI*frame/NFRAMES; break;
				case 1: roty = 6*M_PI*frame/NFRAMES; break;
				case 2: rotz = 6*M_PI*frame/NFRAMES; break;
			}
			computeFrame(frame, rotx, roty, rotz, vol, rot, im, smooth);
			
			MPI_Send(smooth->imdata[0], width*height, MPI_INT, 0, RESULT_TAG,
					MPI_COMM_WORLD);
		}
  
		freeImage(smooth);
		freeVolume(rot);
		
	}
	
	
	pTime = MPI_Wtime()-time; 
	printf("Time of %d: %f\n",rank, pTime);
	
	
	
	if(rank==0) {
		printf("Total Time: %f\n", MPI_Wtime()-time);
		printf("Execution Time:%f\n", extime);
	}
	
	freeVolume(vol);
	freeImage(im);
    MPI_Finalize();

	return EXIT_SUCCESS;
}
