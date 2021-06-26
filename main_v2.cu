#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>
#include<cuda_runtime.h>


namespace Neighbours
{
  
  class FileHandle
  {
  public:

    int InputSize(void)
    {
      FILE *input = NULL;
      input = fopen("input.txt", "r");
  
      char line[30];
      int N = 0;

      while(fgets(line, 30, input) != NULL)
	N++;
  
      fclose(input);
      
      return N;
    }
    
    void ReadFromFile(double *x, double *y, double *z, bool *b, int *N)
    {
      FILE *input = NULL;
      input = fopen("input.txt", "r");

      char line[30];
      
      for(int i = 0; i < (*N); i++)
	{
	  fgets(line, 30, input);
	  sscanf(line, "%lf %lf %lf", &x[i], &y[i], &z[i]);
	  b[i] = true;
	}

      fclose(input);
      
      printf("Data imported from input.txt successfully!\n");
    }

    void WriteToFile(double *x, double *y, double *z, bool *b, int *N)
    {
      FILE *output = NULL;
      output = fopen("output.txt", "w");

      for(int i = 0; i < (*N); i++)
	{
	  if(b[i] == true)
	    fprintf(output, "%.1lf %.1lf %.1lf\n", x[i], y[i], z[i]);
	}

      fclose(output);
      
      printf("Data exported to output.txt successfully!\n");
    }
  };
   
  __global__ void kernel(double *d_xx, double *d_yy, double *d_zz, bool *d_bb, int *d_N, double *x, double *y, double *z, double *r)
  {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
      
    if(index < *d_N)
      {
	if((pow((*x)-d_xx[index], 2) + pow((*y)-d_yy[index], 2) + pow((*z)-d_zz[index], 2)) > pow(*r, 2))
	  d_bb[index] = false;
      }
  }

  class NeighbourSearch
  {
  public:
    void FindNeighbours(double *d_xx, double *d_yy, double *d_zz, bool *d_bb, int *d_N, double *x, double *y, double *z, double *r)
    {
      int grid_size, block_size = 256;
      grid_size = ((*d_N) + block_size) / block_size;
      
      kernel<<<grid_size, block_size>>>(d_xx, d_yy, d_zz, d_bb, d_N, x, y, z, r);

      cudaDeviceSynchronize();
    }

  };   

} // namespace Neighbours

int main()
{
  Neighbours::FileHandle fh = Neighbours::FileHandle();
  double *x, *y, *z;
  double *r;

  double *xx, *yy, *zz;
  bool *bb;
  int *N;

  cudaMallocManaged(&N, sizeof(int));
  *N = fh.InputSize();

  cudaMallocManaged(&x, sizeof(double));
  cudaMallocManaged(&y, sizeof(double));
  cudaMallocManaged(&z, sizeof(double));
  cudaMallocManaged(&r, sizeof(double));
  
  cudaMallocManaged(&xx, sizeof(double)*(*N));
  cudaMallocManaged(&yy, sizeof(double)*(*N));
  cudaMallocManaged(&zz, sizeof(double)*(*N));
  cudaMallocManaged(&bb, sizeof(double)*(*N));   

  fh.ReadFromFile(xx, yy, zz, bb, N);
  
  Neighbours::NeighbourSearch ns = Neighbours::NeighbourSearch();
  
  while(1)
    {
      printf("Enter the x, y and z coordinates of the point and the search distance:\t");
      scanf("%lf %lf %lf %lf", x, y, z, r);

      if((*r) <= 0)
	break;
      else
	{
	  ns.FindNeighbours(xx, yy, zz, bb, N, x, y, z, r);
	  
	  fh.WriteToFile(xx, yy, zz, bb, N);
	}
    }
  
  cudaFree(xx);
  cudaFree(yy);
  cudaFree(zz);
  cudaFree(bb);
  cudaFree(N);

  cudaFree(x);
  cudaFree(y);
  cudaFree(z);
  cudaFree(r);
  
  printf("Program terminated.\n");
  return 0;
}
