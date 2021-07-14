/* File: prime.c
 *A simple program for computing the amount of prime numbers within the 
 *interval [a,b]. */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h> 

#define FALSE 0
#define TRUE  1

#pragma acc routine 
static int isPrime (unsigned int p)
{
	int i, root, out=1;
	if (p == 1)
		return FALSE;
	if (p == 2)
		return TRUE;
	if (p % 2 == 0)
		return FALSE;

	root = (int)(1 + sqrt (p));
	
	for (i = 3; i < root; i += 2)
	  if(p% i ==0) out=0 ;
	  
	return out;
}


int main (int argc, char **argv) 
{
	unsigned int i, a, b, cnt = 0;
	double fstart, fend;
	struct timeval start, end;
		
	system("date"); system("echo $USER");

	fprintf (stdout, "Enter two integer number a, b such that 1<=a<=b: ");
	fflush (stdout);
	scanf ("%u %u", &a, &b);
	
	gettimeofday (&start, NULL);

	if (a <= 2)
	{
		cnt = 1;
		a = 3;
	}
	if (a % 2 == 0)
		a++;
	#pragma acc parallel 
	{
	#pragma acc loop reduction(+:cnt) 
	for (i = a; i <= b; i += 2) {
		if (isPrime (i)) {
			cnt++;
			
		}
	}
    }
	gettimeofday (&end, NULL);
	fstart = (start.tv_sec * 1000000.0 + start.tv_usec) / 1000000.0;
	fend = (end.tv_sec * 1000000.0 + end.tv_usec) / 1000000.0;
	printf ("wallclock: %lf seconds\n", fend-fstart);		

	fprintf (stdout, "\n#primes=%u\n", cnt);
	fflush (stdout);
	return EXIT_SUCCESS;
}
