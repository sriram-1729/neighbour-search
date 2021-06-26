#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>
#include<cuda_runtime.h>


namespace placeholder
{
  
  class Data
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
    
    void ReadFromFile(double *x, double *y, double *z, bool *b, int N)
    {
      FILE *input = NULL;
      input = fopen("input.txt", "r");

      char line[30];
      
      for(int i = 0; i < N; i++)
	{
	  fgets(line, 30, input);
	  sscanf(line, "%lf %lf %lf", &x[i], &y[i], &z[i]);
	  b[i] = true;
	}

      fclose(input);
      
      printf("Data imported from input.txt successfully!\n");
    }

    void WriteToFile(double *x, double *y, double *z, bool *b, int N)
    {
      FILE *output = NULL;
      output = fopen("output.txt", "w");

      for(int i = 0; i < N; i++)
	{
	  if(b[i] == true)
	    fprintf(output, "%.1lf %.1lf %.1lf\n", x[i], y[i], z[i]);
	}

      fclose(output);
      
      printf("Data exported to output.txt successfully!\n");
    }
  };

  class NeighbourSearch
  {
  public:
    
    void CopyFromHostToDevice(double *x1, double *x2, double *y1, double *y2, double *z1, double *z2, bool *b1, bool *b2, int N1, int *N2)
    {
      cudaMemcpy(N2, &N1, sizeof(int), cudaMemcpyHostToDevice);

      cudaMemcpy(x2, x1, sizeof(double)*N1, cudaMemcpyHostToDevice);
      cudaMemcpy(y2, y1, sizeof(double)*N1, cudaMemcpyHostToDevice);
      cudaMemcpy(z2, z1, sizeof(double)*N1, cudaMemcpyHostToDevice);
      cudaMemcpy(b2, b1, sizeof(bool)*N1, cudaMemcpyHostToDevice);
      
    }
    
    void CopyFromDeviceToHost(bool *b1, bool *b2, int N)
    {
      cudaMemcpy(b2, b1, sizeof(bool)*N, cudaMemcpyDeviceToHost);
    }
    

  };
  
  __global__ void search(double *d_xx, double *d_yy, double *d_zz, bool *d_bb, int *d_N, double *x, double *y, double *z, double *r)
  {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < *d_N)
      {
	if((pow((*x)-d_xx[index], 2) + pow((*y)-d_yy[index], 2) + pow((*z)-d_zz[index], 2)) > pow(*r, 2))
	  d_bb[index] = false;
      }
  }
  
}

int main()
{
  placeholder::Data f1 = placeholder::Data();
  double x, y, z;
  double r;
  
  double *d_x, *d_y, *d_z;
  double *d_r;

  double *xx, *yy, *zz;
  bool *bb;
  int N = f1.InputSize();

  double *d_xx, *d_yy, *d_zz;
  bool *d_bb;
  int *d_N;

  int grid_size, block_size = 256;

  xx = (double *)malloc(sizeof(double)*N);
  yy = (double *)malloc(sizeof(double)*N);
  zz = (double *)malloc(sizeof(double)*N);
  bb = (bool *)malloc(sizeof(bool)*N);
  
  f1.ReadFromFile(xx, yy, zz, bb, N);

  cudaMalloc((void **)&d_xx, sizeof(double) * N);
  cudaMalloc((void **)&d_yy, sizeof(double) * N);
  cudaMalloc((void **)&d_zz, sizeof(double) * N);
  cudaMalloc((void **)&d_bb, sizeof(bool) * N);
  cudaMalloc((void **)&d_N, sizeof(int));
	     
  cudaMalloc((void **)&d_x, sizeof(double));
  cudaMalloc((void **)&d_y, sizeof(double));
  cudaMalloc((void **)&d_z, sizeof(double));
  cudaMalloc((void **)&d_r, sizeof(double));
	     
  placeholder::NeighbourSearch ns;

  ns.CopyFromHostToDevice(xx, d_xx, yy, d_yy, zz, d_zz, bb, d_bb, N, d_N);
  
  printf("To end the program, type a number which is not positive in the search distance field.\n");
  
  while(1)
    {
      printf("Enter the x, y and z coordinates of the point and the search distance:\t");
      scanf("%lf %lf %lf %lf", &x, &y, &z, &r);

      if(r <= 0)
	break;
      else
	{
	  cudaMemcpy(d_x, &x, sizeof(double), cudaMemcpyHostToDevice);
	  cudaMemcpy(d_y, &y, sizeof(double), cudaMemcpyHostToDevice);
	  cudaMemcpy(d_z, &z, sizeof(double), cudaMemcpyHostToDevice);
	  cudaMemcpy(d_r, &r, sizeof(double), cudaMemcpyHostToDevice);
	  
	  grid_size = (N + block_size) / block_size;
	  placeholder::search<<<grid_size,block_size>>>(d_xx, d_yy, d_zz, d_bb, d_N, d_x, d_y, d_z, d_r);
	  
	  ns.CopyFromDeviceToHost(d_bb, bb, N);

	  f1.WriteToFile(xx, yy, zz, bb, N);
	}
    }
  
  free(xx);
  free(yy);
  free(zz);
  free(bb);
  cudaFree(d_xx);
  cudaFree(d_yy);
  cudaFree(d_zz);
  cudaFree(d_bb);
  cudaFree(d_N);

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  cudaFree(d_r);

  printf("Program terminated.\n");
  return 0;
}
