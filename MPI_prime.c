// File: prime.c
// 
// A simple program for computing the amount of prime numbers within the 
// interval [a,b].

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#define FALSE 0
#define TRUE  1

static int isPrime (unsigned int p)
{
	int i, root;
	if (p == 1)
		return FALSE;
	if (p == 2)
		return TRUE;
	if (p % 2 == 0)
		return FALSE;

	root = (int)(1 + sqrt (p));
	for (i = 3; (i < root) && (p % i != 0); i += 2);
	return i < root ? FALSE : TRUE;
}

int main (int argc, char **argv)
{		

	unsigned int i, a, b, cnt = 0;
	int size, rank;
	double time;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Status status;
	

	
	int msg[2];
	int range;
	if(rank == 0)
	{
		// Prints current date and user. DO NOT MODIFY
		system("date"); system("echo $USER");
		fprintf (stdout, "Enter two integer number a, b such that 1<=a<=b: ");
		fflush (stdout);
		scanf ("%u %u", &a, &b);
		
		msg[0] = a;
		msg[1] = a + ((b-a)/size);
		range = ((b-a)/size);
		for(i=1; i<size; ++i)
		{
			msg[0] += range + 1;
			msg[1] = (i == size-1 ? b : msg[1]+range+1);
			MPI_Send(msg, sizeof(msg), MPI_INT, i, 0, MPI_COMM_WORLD);
		}
		b = a + range;
	} else {
		MPI_Recv(msg, sizeof(msg), MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		a = msg[0];
		b = msg[1];
	}
	time = MPI_Wtime();
		
	if (a <= 2)
	{
		cnt = 1;
		a = 3;
	}
	
	if (a % 2 == 0)
		a++;
	
	for (i = a; i <= b; i += 2)
		if (isPrime (i)) {
			cnt++;
		}
	printf("Time in %d: %lf\n", rank, MPI_Wtime() - time);
	if(rank > 0)
	{
		MPI_Send(&cnt, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
	}
	
	int sum;
	if(rank == 0)
	{
		sum = cnt; 
		for (i = 1; i<size; i++)
		{ 
			MPI_Recv(&cnt, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
			sum += cnt; 
		}
		fprintf (stdout, "\n#primes=%u\n", sum);
		fflush (stdout);
	}
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}
