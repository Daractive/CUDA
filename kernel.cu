
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <omp.h>
#include <math.h>

#define ZADANIE5

#ifdef ZADANIE1

inline int _ConvertSMVer2Cores(int major, int minor) {
	// Defines for GPU Architecture types (using the SM version to determine
	// the # of cores per SM
	typedef struct {
		int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
		// and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = {
		{0x30, 192},
		{0x32, 192},
		{0x35, 192},
		{0x37, 192},
		{0x50, 128},
		{0x52, 128},
		{0x53, 128},
		{0x60,  64},
		{0x61, 128},
		{0x62, 128},
		{0x70,  64},
		{0x72,  64},
		{0x75,  64},
		{-1, -1} };

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
			return nGpuArchCoresPerSM[index].Cores;
		}

		index++;
	}

	// If we don't find the values, we default use the previous one
	// to run properly
	printf(
		"MapSMtoCores for SM %d.%d is undefined."
		"  Default to use %d Cores/SM\n",
		major, minor, nGpuArchCoresPerSM[index - 1].Cores);
	return nGpuArchCoresPerSM[index - 1].Cores;
}

#endif

#ifdef ZADANIE3

__global__ void addKernel(int *c, int *a, int *b)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void mulKernel(int *c, int *a, int *b)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = a[i] * b[i];
}

__global__ void powKernel(int *c, int *a, int *b)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int result = 1;
	for (int j = 0; j < b[i]; j++)
	{
		result *= a[i];
	}
	c[i] = result;
}

__global__ void addKernel(float *c, float *a, float *b)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void mulKernel(float *c, float *a, float *b)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = a[i] * b[i];
}

__global__ void powKernel(float *c, float *a, float *b)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float result = 1;
	for (int j = 0; j < b[i]; j++)
	{
		result *= a[i];
	}
	c[i] = result;
}

__global__ void addKernel(double *c, double *a, double *b)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void mulKernel(double *c, double *a, double *b)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = a[i] * b[i];
}

__global__ void powKernel(double *c, double *a, double *b)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	double result = 1;
	for (int j = 0; j < b[i]; j++)
	{
		result *= a[i];
	}
	c[i] = result;
}

void addCPU(int *c, int *a, int *b, int size);
void mulCPU(int *c, int *a, int *b, int size);
void powCPU(int *c, int *a, int *b, int size);
void addCPU(float *c, float *a, float *b, int size);
void mulCPU(float *c, float *a, float *b, int size);
void powCPU(float *c, float *a, float *b, int size);
void addCPU(double *c, double *a, double *b, int size);
void mulCPU(double *c, double *a, double *b, int size);
void powCPU(double *c, double *a, double *b, int size);

#endif

#ifdef ZADANIE4

__global__ void addMatrixKernel(float *c, float *a, float *b, int n)
{
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n && j < n) {
		c[i * n + j] = a[i * n + j] + b[i * n + j];
	}
}
__global__ void mulMatrixKernel(float *c, float *a, float *b, int n)
{
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	float temp = 0;
	if (i < n && j < n) {
		for (int k = 0; k < n; k++)
		{
			temp += a[i * n + k] * b[k * n + j];
		}
	}
	c[i * n + j] = temp;
}
__global__ void addMatrixKernel(double *c, double *a, double *b, int n)
{
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n && j < n) {
		c[i * n + j] = a[i * n + j] + b[i * n + j];
	}
}
__global__ void mulMatrixKernel(double *c, double *a, double *b, int n)
{
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	double temp = 0;
	if (i < n && j < n) {
		for (int k = 0; k < n; k++)
		{
			temp += a[i * n + k] * b[k * n + j];
		}
	}
	c[i * n + j] = temp;
}

void addMatrixCPU(float *c, float *a, float *b, int size);
void mulMatrixCPU(float *c, float *a, float *b, int size);
void addMatrixCPU(double *c, double *a, double *b, int size);
void mulMatrixCPU(double *c, double *a, double *b, int size);

#endif

#ifdef ZADANIE5

__global__ void addMatrixKernel(float *c, cudaTextureObject_t tex, cudaTextureObject_t tex2, int n)
{
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	float a = tex1Dfetch<float>(tex, i * n + j);
	float b = tex1Dfetch<float>(tex2, i * n + j);
	if (i < n && j < n) {
		c[i * n + j] = a + b;
	}
}
__global__ void mulMatrixKernel(float *c, cudaTextureObject_t tex, cudaTextureObject_t tex2, int n)
{
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	float temp = 0, a, b;
	if (i < n && j < n) {
		for (int k = 0; k < n; k++)
		{
			a = tex1Dfetch<float>(tex, i * n + k);
			b = tex1Dfetch<float>(tex2, k * n + j);
			temp += a * b;
		}
	}
	c[i * n + j] = temp;
}
#endif

int main()
{

#ifdef ZADANIE1

	cudaDeviceProp prop;
	int count = 0;
	cudaGetDeviceCount(&count);
	cudaGetDeviceProperties(&prop, 0);
	printf("Dane urz¹dzenia: %d\n", 0);
	printf("Numer: %d\n", 0);
	printf("Nazwa: %s\n", prop.name);
	printf("Liczba rdzeni: %d\n", (_ConvertSMVer2Cores(prop.major, prop.minor)) * prop.multiProcessorCount);
	printf("Liczba multiprocesorow: %d\n", prop.multiProcessorCount);
	printf("Wsparcie CUDA COMPUTE COMPATIBILITY: %d %d\n", prop.major, prop.minor);

	cudaSetDevice(0);

	cudaDeviceReset();

#endif ZADANIE1

#ifdef ZADANIE2


	char *MiB1charGPU, *MiB8charGPU, *MiB96charGPU, *MiB256charGPU;
	char *MiB1charCPU, *MiB8charCPU, *MiB96charCPU, *MiB256charCPU;
	int *MiB1intGPU, *MiB8intGPU, *MiB96intGPU, *MiB256intGPU;
	int *MiB1intCPU, *MiB8intCPU, *MiB96intCPU, *MiB256intCPU;
	float *MiB1floatGPU, *MiB8floatGPU, *MiB96floatGPU, *MiB256floatGPU;
	float *MiB1floatCPU, *MiB8floatCPU, *MiB96floatCPU, *MiB256floatCPU;
	double *MiB1doubleGPU, *MiB8doubleGPU, *MiB96doubleGPU, *MiB256doubleGPU;
	double *MiB1doubleCPU, *MiB8doubleCPU, *MiB96doubleCPU, *MiB256doubleCPU;

	cudaSetDevice(0);

	cudaEvent_t start, stop;
	float timer = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	{
		MiB1charCPU = (char*)malloc(1024 * 1024);
		MiB8charCPU = (char*)malloc(8 * 1024 * 1024);
		MiB96charCPU = (char*)malloc(96 * 1024 * 1024);
		MiB256charCPU = (char*)malloc(256 * 1024 * 1024);

		cudaMalloc(&MiB1charGPU, 1024 * 1024);
		cudaMalloc(&MiB8charGPU, 8 * 1024 * 1024);
		cudaMalloc(&MiB96charGPU, 96 * 1024 * 1024);
		cudaMalloc(&MiB256charGPU, 256 * 1024 * 1024);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB1charGPU, MiB1charCPU, 1024 * 1024, cudaMemcpyHostToDevice); //CPU to GPU
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 1 MiB char z CPU do GPU w ms: %f\n", timer);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB8charGPU, MiB8charCPU, 8 * 1024 * 1024, cudaMemcpyHostToDevice);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 8 MiB char z CPU do GPU w ms: %f\n", timer);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB96charGPU, MiB96charCPU, 96 * 1024 * 1024, cudaMemcpyHostToDevice);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 96 MiB char z CPU do GPU w ms: %f\n", timer);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB256charGPU, MiB256charCPU, 256 * 1024 * 1024, cudaMemcpyHostToDevice);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 256 MiB char z CPU do GPU w ms: %f\n", timer);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB1charCPU, MiB1charGPU, 1024 * 1024, cudaMemcpyDeviceToHost); //GPU to CPU
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 1 MiB char z GPU do CPU w ms: %f\n", timer);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB8charCPU, MiB8charGPU, 8 * 1024 * 1024, cudaMemcpyDeviceToHost);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 8 MiB char z GPU do CPU w ms: %f\n", timer);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB96charCPU, MiB96charGPU, 96 * 1024 * 1024, cudaMemcpyDeviceToHost);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 96 MiB char z GPU do CPU w ms: %f\n", timer);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB256charCPU, MiB256charGPU, 256 * 1024 * 1024, cudaMemcpyDeviceToHost);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 256 MiB char z GPU do CPU w ms: %f\n", timer);

		cudaFree(MiB1charGPU);
		cudaFree(MiB8charGPU);
		cudaFree(MiB96charGPU);
		cudaFree(MiB256charGPU);

		free(MiB1charCPU);
		free(MiB8charCPU);
		free(MiB96charCPU);
		free(MiB256charCPU);
	}

	{
		MiB1intCPU = (int*)malloc(1024 * 1024);
		MiB8intCPU = (int*)malloc(8 * 1024 * 1024);
		MiB96intCPU = (int*)malloc(96 * 1024 * 1024);
		MiB256intCPU = (int*)malloc(256 * 1024 * 1024);

		cudaMalloc(&MiB1intGPU, 1024 * 1024);
		cudaMalloc(&MiB8intGPU, 8 * 1024 * 1024);
		cudaMalloc(&MiB96intGPU, 96 * 1024 * 1024);
		cudaMalloc(&MiB256intGPU, 256 * 1024 * 1024);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB1intGPU, MiB1intCPU, 1024 * 1024, cudaMemcpyHostToDevice);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 1 MiB int z CPU do GPU w ms: %f\n", timer);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB8intGPU, MiB8intCPU, 8 * 1024 * 1024, cudaMemcpyHostToDevice);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 8 MiB int z CPU do GPU w ms: %f\n", timer);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB96intGPU, MiB96intCPU, 96 * 1024 * 1024, cudaMemcpyHostToDevice);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 96 MiB int z CPU do GPU w ms: %f\n", timer);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB256intGPU, MiB256intCPU, 256 * 1024 * 1024, cudaMemcpyHostToDevice);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 256 MiB int z CPU do GPU w ms: %f\n", timer);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB1intCPU, MiB1intGPU, 1024 * 1024, cudaMemcpyDeviceToHost);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 1 MiB int z GPU do CPU w ms: %f\n", timer);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB8intCPU, MiB8intGPU, 8 * 1024 * 1024, cudaMemcpyDeviceToHost);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 8 MiB int z GPU do CPU w ms: %f\n", timer);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB96intCPU, MiB96intGPU, 96 * 1024 * 1024, cudaMemcpyDeviceToHost);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 96 MiB int z GPU do CPU w ms: %f\n", timer);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB256intCPU, MiB256intGPU, 256 * 1024 * 1024, cudaMemcpyDeviceToHost);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 256 MiB int z GPU do CPU w ms: %f\n", timer);

		cudaFree(MiB1intGPU);
		cudaFree(MiB8intGPU);
		cudaFree(MiB96intGPU);
		cudaFree(MiB256intGPU);

		free(MiB1intCPU);
		free(MiB8intCPU);
		free(MiB96intCPU);
		free(MiB256intCPU);
	}

	{
		MiB1floatCPU = (float*)malloc(1024 * 1024);
		MiB8floatCPU = (float*)malloc(8 * 1024 * 1024);
		MiB96floatCPU = (float*)malloc(96 * 1024 * 1024);
		MiB256floatCPU = (float*)malloc(256 * 1024 * 1024);

		cudaMalloc(&MiB1floatGPU, 1024 * 1024);
		cudaMalloc(&MiB8floatGPU, 8 * 1024 * 1024);
		cudaMalloc(&MiB96floatGPU, 96 * 1024 * 1024);
		cudaMalloc(&MiB256floatGPU, 256 * 1024 * 1024);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB1floatGPU, MiB1floatCPU, 1024 * 1024, cudaMemcpyHostToDevice);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 1 MiB float z CPU do GPU w ms: %f\n", timer);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB8floatGPU, MiB8floatCPU, 8 * 1024 * 1024, cudaMemcpyHostToDevice);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 8 MiB float z CPU do GPU w ms: %f\n", timer);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB96floatGPU, MiB96floatCPU, 96 * 1024 * 1024, cudaMemcpyHostToDevice);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 96 MiB float z CPU do GPU w ms: %f\n", timer);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB256floatGPU, MiB256floatCPU, 256 * 1024 * 1024, cudaMemcpyHostToDevice);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 256 MiB float z CPU do GPU w ms: %f\n", timer);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB1floatCPU, MiB1floatGPU, 1024 * 1024, cudaMemcpyDeviceToHost);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 1 MiB float z GPU do CPU w ms: %f\n", timer);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB8floatCPU, MiB8floatGPU, 8 * 1024 * 1024, cudaMemcpyDeviceToHost);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 8 MiB float z GPU do CPU w ms: %f\n", timer);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB96floatCPU, MiB96floatGPU, 96 * 1024 * 1024, cudaMemcpyDeviceToHost);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 96 MiB float z GPU do CPU w ms: %f\n", timer);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB256floatCPU, MiB256floatGPU, 256 * 1024 * 1024, cudaMemcpyDeviceToHost);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 256 MiB float z GPU do CPU w ms: %f\n", timer);

		cudaFree(MiB1floatGPU);
		cudaFree(MiB8floatGPU);
		cudaFree(MiB96floatGPU);
		cudaFree(MiB256floatGPU);

		free(MiB1floatCPU);
		free(MiB8floatCPU);
		free(MiB96floatCPU);
		free(MiB256floatCPU);
	}

	{
		MiB1doubleCPU = (double*)malloc(1024 * 1024);
		MiB8doubleCPU = (double*)malloc(8 * 1024 * 1024);
		MiB96doubleCPU = (double*)malloc(96 * 1024 * 1024);
		MiB256doubleCPU = (double*)malloc(256 * 1024 * 1024);

		cudaMalloc(&MiB1doubleGPU, 1024 * 1024);
		cudaMalloc(&MiB8doubleGPU, 8 * 1024 * 1024);
		cudaMalloc(&MiB96doubleGPU, 96 * 1024 * 1024);
		cudaMalloc(&MiB256doubleGPU, 256 * 1024 * 1024);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB1doubleGPU, MiB1doubleCPU, 1024 * 1024, cudaMemcpyHostToDevice);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 1 MiB double z CPU do GPU: %f\n", timer);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB8doubleGPU, MiB8doubleCPU, 8 * 1024 * 1024, cudaMemcpyHostToDevice);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 8 MiB double z CPU do GPU: %f\n", timer);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB96doubleGPU, MiB96doubleCPU, 96 * 1024 * 1024, cudaMemcpyHostToDevice);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 96 MiB double z CPU do GPU: %f\n", timer);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB256doubleGPU, MiB256doubleCPU, 256 * 1024 * 1024, cudaMemcpyHostToDevice);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 256 MiB double z CPU do GPU: %f\n", timer);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB1doubleCPU, MiB1doubleGPU, 1024 * 1024, cudaMemcpyDeviceToHost);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 1 MiB double z GPU do CPU: %f\n", timer);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB8doubleCPU, MiB8doubleGPU, 8 * 1024 * 1024, cudaMemcpyDeviceToHost);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 8 MiB double z GPU do CPU: %f\n", timer);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB96doubleCPU, MiB96doubleGPU, 96 * 1024 * 1024, cudaMemcpyDeviceToHost);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 96 MiB double z GPU do CPU: %f\n", timer);

		cudaEventRecord(start, 0);
		cudaMemcpy(MiB256doubleCPU, MiB256doubleGPU, 256 * 1024 * 1024, cudaMemcpyDeviceToHost);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas przesy³ania 256 MiB double z GPU do CPU: %f\n", timer);

		cudaFree(MiB1doubleGPU);
		cudaFree(MiB8doubleGPU);
		cudaFree(MiB96doubleGPU);
		cudaFree(MiB256doubleGPU);

		free(MiB1doubleCPU);
		free(MiB8doubleCPU);
		free(MiB96doubleCPU);
		free(MiB256doubleCPU);
	}

	cudaDeviceReset();

#endif ZADANIE2

#ifdef ZADANIE3

	cudaSetDevice(0);

	{
		const int size = 10;
		int a[size] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
		int b[size] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
		int c[size] = { 0 };
		int *dev_a;
		int *dev_b;
		int *dev_c;

		cudaMalloc(&dev_a, size * sizeof(int));
		cudaMalloc(&dev_b, size * sizeof(int));
		cudaMalloc(&dev_c, size * sizeof(int));

		cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_c, c, size * sizeof(int), cudaMemcpyHostToDevice);

		addKernel << <1, size >> > (dev_c, dev_a, dev_b);
		cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
		printf("Dodawanie GPU:\n");
		printf("{%d,%d,%d,%d,%d,%d,%d,%d,%d,%d} + {%d,%d,%d,%d,%d,%d,%d,%d,%d,%d} = {%d,%d,%d,%d,%d,%d,%d,%d,%d,%d}\n",
			a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9],
			b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9],
			c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]);

		mulKernel << <1, size >> > (dev_c, dev_a, dev_b);
		cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
		printf("Mno¿enie GPU:\n");
		printf("{%d,%d,%d,%d,%d,%d,%d,%d,%d,%d} * {%d,%d,%d,%d,%d,%d,%d,%d,%d,%d} = {%d,%d,%d,%d,%d,%d,%d,%d,%d,%d}\n",
			a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9],
			b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9],
			c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]);

		powKernel << <1, size >> > (dev_c, dev_a, dev_b);
		cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
		printf("Potêgowanie GPU:\n");
		printf("{%d,%d,%d,%d,%d,%d,%d,%d,%d,%d} ^ {%d,%d,%d,%d,%d,%d,%d,%d,%d,%d} = {%d,%d,%d,%d,%d,%d,%d,%d,%d,%d}\n",
			a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9],
			b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9],
			c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]);

		addCPU(c, a, b, size);
		printf("Dodawanie CPU:\n");
		printf("{%d,%d,%d,%d,%d,%d,%d,%d,%d,%d} + {%d,%d,%d,%d,%d,%d,%d,%d,%d,%d} = {%d,%d,%d,%d,%d,%d,%d,%d,%d,%d}\n",
			a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9],
			b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9],
			c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]);

		mulCPU(c, a, b, size);
		printf("Mno¿enie CPU:\n");
		printf("{%d,%d,%d,%d,%d,%d,%d,%d,%d,%d} * {%d,%d,%d,%d,%d,%d,%d,%d,%d,%d} = {%d,%d,%d,%d,%d,%d,%d,%d,%d,%d}\n",
			a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9],
			b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9],
			c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]);

		powCPU(c, a, b, size);
		printf("Potêgowanie CPU:\n");
		printf("{%d,%d,%d,%d,%d,%d,%d,%d,%d,%d} ^ {%d,%d,%d,%d,%d,%d,%d,%d,%d,%d} = {%d,%d,%d,%d,%d,%d,%d,%d,%d,%d}\n",
			a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9],
			b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9],
			c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]);

		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);
	}

	int *MiB1intGPU, *MiB4intGPU, *MiB8intGPU, *MiB16intGPU, *resIntGPU;
	int *MiB1intCPU, *MiB4intCPU, *MiB8intCPU, *MiB16intCPU, *resIntCPU;
	float *MiB1floatGPU, *MiB4floatGPU, *MiB8floatGPU, *MiB16floatGPU, *resFloatGPU;
	float *MiB1floatCPU, *MiB4floatCPU, *MiB8floatCPU, *MiB16floatCPU, *resFloatCPU;
	double *MiB1doubleGPU, *MiB4doubleGPU, *MiB8doubleGPU, *MiB16doubleGPU, *resDoubleGPU;
	double *MiB1doubleCPU, *MiB4doubleCPU, *MiB8doubleCPU, *MiB16doubleCPU, *resDoubleCPU;

	cudaSetDevice(0);

	cudaEvent_t start, stop;
	float timer = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	double startOMP;
	double stopOMP;
	int blockSize = 256;
	int size;
	int numBlocks;

	{
		MiB1intCPU = (int*)malloc(1024 * 1024);
		MiB4intCPU = (int*)malloc(4 * 1024 * 1024);
		MiB8intCPU = (int*)malloc(8 * 1024 * 1024);
		MiB16intCPU = (int*)malloc(16 * 1024 * 1024);
		resIntCPU = (int*)malloc(16 * 1024 * 1024);

		cudaMalloc(&MiB1intGPU, 1024 * 1024);
		cudaMalloc(&MiB4intGPU, 4 * 1024 * 1024);
		cudaMalloc(&MiB8intGPU, 8 * 1024 * 1024);
		cudaMalloc(&MiB16intGPU, 16 * 1024 * 1024);
		cudaMalloc(&resIntGPU, 16 * 1024 * 1024);

		size = 1024 * 1024 / sizeof(int);
		numBlocks = (size + blockSize - 1) / blockSize;
		cudaEventRecord(start, 0);
		addKernel << <numBlocks, blockSize >> > (resIntGPU, MiB1intGPU, MiB1intGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas dodawania 1 MiB int na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		addCPU(resIntCPU, MiB1intCPU, MiB1intCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas dodawania 1 MiB int na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		mulKernel << <numBlocks, blockSize >> > (resIntGPU, MiB1intGPU, MiB1intGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas mno¿enia 1 MiB int na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		mulCPU(resIntCPU, MiB1intCPU, MiB1intCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas mno¿enia 1 MiB int na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		powKernel << <numBlocks, blockSize >> > (resIntGPU, MiB1intGPU, MiB1intGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas potêgowania 1 MiB int na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		mulCPU(resIntCPU, MiB1intCPU, MiB1intCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas potêgowania 1 MiB int na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		size = 4 * 1024 * 1024 / sizeof(int);
		numBlocks = (size + blockSize - 1) / blockSize;
		cudaEventRecord(start, 0);
		addKernel << <numBlocks, blockSize >> > (resIntGPU, MiB4intGPU, MiB4intGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas dodawania 4 MiB int na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		addCPU(resIntCPU, MiB4intCPU, MiB4intCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas dodawania 4 MiB int na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		mulKernel << <numBlocks, blockSize >> > (resIntGPU, MiB4intGPU, MiB4intGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas mno¿enia 4 MiB int na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		mulCPU(resIntCPU, MiB4intCPU, MiB4intCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas mno¿enia 4 MiB int na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		powKernel << <numBlocks, blockSize >> > (resIntGPU, MiB4intGPU, MiB4intGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas potêgowania 4 MiB int na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		powCPU(resIntCPU, MiB4intCPU, MiB4intCPU, size);
		stopOMP = omp_get_wtime();
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas potêgowania 4 MiB int na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		size = 8 * 1024 * 1024 / sizeof(int);
		numBlocks = (size + blockSize - 1) / blockSize;
		cudaEventRecord(start, 0);
		addKernel << <numBlocks, blockSize >> > (resIntGPU, MiB8intGPU, MiB8intGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas dodawania 8 MiB int na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		addCPU(resIntCPU, MiB8intCPU, MiB8intCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas dodawania 8 MiB int na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		mulKernel << <numBlocks, blockSize >> > (resIntGPU, MiB8intGPU, MiB8intGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas mno¿enia 8 MiB int na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		mulCPU(resIntCPU, MiB8intCPU, MiB8intCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas mno¿enia 8 MiB int na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		powKernel << <numBlocks, blockSize >> > (resIntGPU, MiB8intGPU, MiB8intGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas potêgowania 8 MiB int na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		powCPU(resIntCPU, MiB8intCPU, MiB8intCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas potêgowania 8 MiB int na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		size = 16 * 1024 * 1024 / sizeof(int);
		numBlocks = (size + blockSize - 1) / blockSize;
		cudaEventRecord(start, 0);
		addKernel << <numBlocks, blockSize >> > (resIntGPU, MiB16intGPU, MiB16intGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas dodawania 16 MiB int na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		addCPU(resIntCPU, MiB16intCPU, MiB16intCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas dodawania 16 MiB int na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		mulKernel << <numBlocks, blockSize >> > (resIntGPU, MiB16intGPU, MiB16intGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas mno¿enia 16 MiB int na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		mulCPU(resIntCPU, MiB16intCPU, MiB16intCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas mno¿enia 16 MiB int na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		powKernel << <numBlocks, blockSize >> > (resIntGPU, MiB16intGPU, MiB16intGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas potêgowania 16 MiB int na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		powCPU(resIntCPU, MiB16intCPU, MiB16intCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas potêgowania 16 MiB int na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaFree(MiB1intGPU);
		cudaFree(MiB4intGPU);
		cudaFree(MiB8intGPU);
		cudaFree(MiB16intGPU);
		cudaFree(resIntGPU);

		free(MiB1intCPU);
		free(MiB4intCPU);
		free(MiB8intCPU);
		free(MiB16intCPU);
		free(resIntCPU);
	}

	{
		MiB1floatCPU = (float*)malloc(1024 * 1024);
		MiB4floatCPU = (float*)malloc(4 * 1024 * 1024);
		MiB8floatCPU = (float*)malloc(8 * 1024 * 1024);
		MiB16floatCPU = (float*)malloc(16 * 1024 * 1024);
		resFloatCPU = (float*)malloc(16 * 1024 * 1024);

		cudaMalloc(&MiB1floatGPU, 1024 * 1024);
		cudaMalloc(&MiB4floatGPU, 4 * 1024 * 1024);
		cudaMalloc(&MiB8floatGPU, 8 * 1024 * 1024);
		cudaMalloc(&MiB16floatGPU, 16 * 1024 * 1024);
		cudaMalloc(&resFloatGPU, 16 * 1024 * 1024);

		size = 1024 * 1024 / sizeof(float);
		numBlocks = (size + blockSize - 1) / blockSize;
		cudaEventRecord(start, 0);
		addKernel << <numBlocks, blockSize >> > (resFloatGPU, MiB1floatGPU, MiB1floatGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas dodawania 1 MiB float na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		addCPU(resFloatCPU, MiB1floatCPU, MiB1floatCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas dodawania 1 MiB float na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		mulKernel << <numBlocks, blockSize >> > (resFloatGPU, MiB1floatGPU, MiB1floatGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas mno¿enia 1 MiB float na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		mulCPU(resFloatCPU, MiB1floatCPU, MiB1floatCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas mno¿enia 1 MiB float na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		powKernel << <numBlocks, blockSize >> > (resFloatGPU, MiB1floatGPU, MiB1floatGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas potêgowania 1 MiB float na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		mulCPU(resFloatCPU, MiB1floatCPU, MiB1floatCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas potêgowania 1 MiB float na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		size = 4 * 1024 * 1024 / sizeof(float);
		numBlocks = (size + blockSize - 1) / blockSize;
		cudaEventRecord(start, 0);
		addKernel << <numBlocks, blockSize >> > (resFloatGPU, MiB4floatGPU, MiB4floatGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas dodawania 4 MiB float na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		addCPU(resFloatCPU, MiB4floatCPU, MiB4floatCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas dodawania 4 MiB float na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		mulKernel << <numBlocks, blockSize >> > (resFloatGPU, MiB4floatGPU, MiB4floatGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas mno¿enia 4 MiB float na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		mulCPU(resFloatCPU, MiB4floatCPU, MiB4floatCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas mno¿enia 4 MiB float na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		powKernel << <numBlocks, blockSize >> > (resFloatGPU, MiB4floatGPU, MiB4floatGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas potêgowania 4 MiB float na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		powCPU(resFloatCPU, MiB4floatCPU, MiB4floatCPU, size);
		stopOMP = omp_get_wtime();
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas potêgowania 4 MiB float na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		size = 8 * 1024 * 1024 / sizeof(float);
		numBlocks = (size + blockSize - 1) / blockSize;
		cudaEventRecord(start, 0);
		addKernel << <numBlocks, blockSize >> > (resFloatGPU, MiB8floatGPU, MiB8floatGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas dodawania 8 MiB float na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		addCPU(resFloatCPU, MiB8floatCPU, MiB8floatCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas dodawania 8 MiB float na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		mulKernel << <numBlocks, blockSize >> > (resFloatGPU, MiB8floatGPU, MiB8floatGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas mno¿enia 8 MiB float na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		mulCPU(resFloatCPU, MiB8floatCPU, MiB8floatCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas mno¿enia 8 MiB float na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		powKernel << <numBlocks, blockSize >> > (resFloatGPU, MiB8floatGPU, MiB8floatGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas potêgowania 8 MiB float na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		powCPU(resFloatCPU, MiB8floatCPU, MiB8floatCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas potêgowania 8 MiB float na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		size = 16 * 1024 * 1024 / sizeof(float);
		numBlocks = (size + blockSize - 1) / blockSize;
		cudaEventRecord(start, 0);
		addKernel << <numBlocks, blockSize >> > (resFloatGPU, MiB16floatGPU, MiB16floatGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas dodawania 16 MiB float na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		addCPU(resFloatCPU, MiB16floatCPU, MiB16floatCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas dodawania 16 MiB float na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		mulKernel << <numBlocks, blockSize >> > (resFloatGPU, MiB16floatGPU, MiB16floatGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas mno¿enia 16 MiB float na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		mulCPU(resFloatCPU, MiB16floatCPU, MiB16floatCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas mno¿enia 16 MiB float na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		powKernel << <numBlocks, blockSize >> > (resFloatGPU, MiB16floatGPU, MiB16floatGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas potêgowania 16 MiB float na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		powCPU(resFloatCPU, MiB16floatCPU, MiB16floatCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas potêgowania 16 MiB float na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaFree(MiB1floatGPU);
		cudaFree(MiB4floatGPU);
		cudaFree(MiB8floatGPU);
		cudaFree(MiB16floatGPU);
		cudaFree(resFloatGPU);

		free(MiB1floatCPU);
		free(MiB4floatCPU);
		free(MiB8floatCPU);
		free(MiB16floatCPU);
		free(resFloatCPU);
	}

	{
		MiB1doubleCPU = (double*)malloc(1024 * 1024);
		MiB4doubleCPU = (double*)malloc(4 * 1024 * 1024);
		MiB8doubleCPU = (double*)malloc(8 * 1024 * 1024);
		MiB16doubleCPU = (double*)malloc(16 * 1024 * 1024);
		resDoubleCPU = (double*)malloc(16 * 1024 * 1024);

		cudaMalloc(&MiB1doubleGPU, 1024 * 1024);
		cudaMalloc(&MiB4doubleGPU, 4 * 1024 * 1024);
		cudaMalloc(&MiB8doubleGPU, 8 * 1024 * 1024);
		cudaMalloc(&MiB16doubleGPU, 16 * 1024 * 1024);
		cudaMalloc(&resDoubleGPU, 16 * 1024 * 1024);

		size = 1024 * 1024 / sizeof(double);
		numBlocks = (size + blockSize - 1) / blockSize;
		cudaEventRecord(start, 0);
		addKernel << <numBlocks, blockSize >> > (resDoubleGPU, MiB1doubleGPU, MiB1doubleGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas dodawania 1 MiB double na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		addCPU(resDoubleCPU, MiB1doubleCPU, MiB1doubleCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas dodawania 1 MiB double na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		mulKernel << <numBlocks, blockSize >> > (resDoubleGPU, MiB1doubleGPU, MiB1doubleGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas mno¿enia 1 MiB double na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		mulCPU(resDoubleCPU, MiB1doubleCPU, MiB1doubleCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas mno¿enia 1 MiB double na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		powKernel << <numBlocks, blockSize >> > (resDoubleGPU, MiB1doubleGPU, MiB1doubleGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas potêgowania 1 MiB double na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		mulCPU(resDoubleCPU, MiB1doubleCPU, MiB1doubleCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas potêgowania 1 MiB double na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		size = 4 * 1024 * 1024 / sizeof(double);
		numBlocks = (size + blockSize - 1) / blockSize;
		cudaEventRecord(start, 0);
		addKernel << <numBlocks, blockSize >> > (resDoubleGPU, MiB4doubleGPU, MiB4doubleGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas dodawania 4 MiB double na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		addCPU(resDoubleCPU, MiB4doubleCPU, MiB4doubleCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas dodawania 4 MiB double na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		mulKernel << <numBlocks, blockSize >> > (resDoubleGPU, MiB4doubleGPU, MiB4doubleGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas mno¿enia 4 MiB double na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		mulCPU(resDoubleCPU, MiB4doubleCPU, MiB4doubleCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas mno¿enia 4 MiB double na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		powKernel << <numBlocks, blockSize >> > (resDoubleGPU, MiB4doubleGPU, MiB4doubleGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas potêgowania 4 MiB double na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		powCPU(resDoubleCPU, MiB4doubleCPU, MiB4doubleCPU, size);
		stopOMP = omp_get_wtime();
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas potêgowania 4 MiB double na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		size = 8 * 1024 * 1024 / sizeof(double);
		numBlocks = (size + blockSize - 1) / blockSize;
		cudaEventRecord(start, 0);
		addKernel << <numBlocks, blockSize >> > (resDoubleGPU, MiB8doubleGPU, MiB8doubleGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas dodawania 8 MiB double na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		addCPU(resDoubleCPU, MiB8doubleCPU, MiB8doubleCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas dodawania 8 MiB double na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		mulKernel << <numBlocks, blockSize >> > (resDoubleGPU, MiB8doubleGPU, MiB8doubleGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas mno¿enia 8 MiB double na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		mulCPU(resDoubleCPU, MiB8doubleCPU, MiB8doubleCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas mno¿enia 8 MiB double na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		powKernel << <numBlocks, blockSize >> > (resDoubleGPU, MiB8doubleGPU, MiB8doubleGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas potêgowania 8 MiB double na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		powCPU(resDoubleCPU, MiB8doubleCPU, MiB8doubleCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas potêgowania 8 MiB double na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		size = 16 * 1024 * 1024 / sizeof(double);
		numBlocks = (size + blockSize - 1) / blockSize;
		cudaEventRecord(start, 0);
		addKernel << <numBlocks, blockSize >> > (resDoubleGPU, MiB16doubleGPU, MiB16doubleGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas dodawania 16 MiB double na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		addCPU(resDoubleCPU, MiB16doubleCPU, MiB16doubleCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas dodawania 16 MiB double na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		mulKernel << <numBlocks, blockSize >> > (resDoubleGPU, MiB16doubleGPU, MiB16doubleGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas mno¿enia 16 MiB double na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		mulCPU(resDoubleCPU, MiB16doubleCPU, MiB16doubleCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas mno¿enia 16 MiB double na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		powKernel << <numBlocks, blockSize >> > (resDoubleGPU, MiB16doubleGPU, MiB16doubleGPU);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas potêgowania 16 MiB double na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		powCPU(resDoubleCPU, MiB16doubleCPU, MiB16doubleCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas potêgowania 16 MiB double na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaFree(MiB1doubleGPU);
		cudaFree(MiB4doubleGPU);
		cudaFree(MiB8doubleGPU);
		cudaFree(MiB16doubleGPU);
		cudaFree(resDoubleGPU);

		free(MiB1doubleCPU);
		free(MiB4doubleCPU);
		free(MiB8doubleCPU);
		free(MiB16doubleCPU);
		free(resDoubleCPU);
	}

	cudaDeviceReset();

#endif

#ifdef ZADANIE4

	float *MiB1floatGPU, *MiB4floatGPU, *MiB8floatGPU, *MiB16floatGPU, *resFloatGPU;
	float *MiB1floatCPU, *MiB4floatCPU, *MiB8floatCPU, *MiB16floatCPU, *resFloatCPU;
	double *MiB1doubleGPU, *MiB4doubleGPU, *MiB8doubleGPU, *MiB16doubleGPU, *resDoubleGPU;
	double *MiB1doubleCPU, *MiB4doubleCPU, *MiB8doubleCPU, *MiB16doubleCPU, *resDoubleCPU;

	cudaSetDevice(0);

	cudaEvent_t start, stop;
	float timer = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	double startOMP;
	double stopOMP;
	int blockSizeX = 32;
	int blockSizeY = 32;
	int size;
	int n;
	int numBlocks;
	dim3 threads(blockSizeX, blockSizeY);

	{
		MiB1floatCPU = (float*)malloc(1024 * 1024);
		MiB4floatCPU = (float*)malloc(4 * 1024 * 1024);
		MiB8floatCPU = (float*)malloc(8 * 1024 * 1024);
		MiB16floatCPU = (float*)malloc(16 * 1024 * 1024);
		resFloatCPU = (float*)malloc(16 * 1024 * 1024);

		cudaMalloc(&MiB1floatGPU, 1024 * 1024);
		cudaMalloc(&MiB4floatGPU, 4 * 1024 * 1024);
		cudaMalloc(&MiB8floatGPU, 8 * 1024 * 1024);
		cudaMalloc(&MiB16floatGPU, 16 * 1024 * 1024);
		cudaMalloc(&resFloatGPU, 16 * 1024 * 1024);

		size = 1024 * 1024 / sizeof(float);
		n = floor(sqrt(size));
		numBlocks = ceil(sqrt((size + blockSizeX * blockSizeY - 1) / (blockSizeX * blockSizeY)));
		dim3 blocks1(numBlocks, numBlocks);
		cudaEventRecord(start, 0);
		addMatrixKernel << <blocks1, threads >> > (resFloatGPU, MiB1floatGPU, MiB1floatGPU, n);
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas dodawania macierzy 1MiB float na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		addMatrixCPU(resFloatCPU, MiB1floatCPU, MiB1floatCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas dodawania macierzy 1MiB float na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		mulMatrixKernel << <blocks1, threads >> > (resFloatGPU, MiB1floatGPU, MiB1floatGPU, n);
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas mno¿enia macierzy 1MiB float na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		mulMatrixCPU(resFloatCPU, MiB1floatCPU, MiB1floatCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas mno¿enia macierzy 1MiB float na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		size = 4 * 1024 * 1024 / sizeof(float);
		n = floor(sqrt(size));
		numBlocks = ceil(sqrt((size + blockSizeX * blockSizeY - 1) / (blockSizeX * blockSizeY)));
		dim3 blocks4(numBlocks, numBlocks);
		cudaEventRecord(start, 0);
		addMatrixKernel << <blocks4, threads >> > (resFloatGPU, MiB4floatGPU, MiB4floatGPU, n);
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas dodawania macierzy 4MiB float na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		addMatrixCPU(resFloatCPU, MiB4floatCPU, MiB4floatCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas dodawania macierzy 4MiB float na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		mulMatrixKernel << <blocks4, threads >> > (resFloatGPU, MiB4floatGPU, MiB4floatGPU, n);
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas mno¿enia macierzy 4MiB float na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		mulMatrixCPU(resFloatCPU, MiB4floatCPU, MiB4floatCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas mno¿enia macierzy 4MiB float  na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		size = 8 * 1024 * 1024 / sizeof(float);
		n = floor(sqrt(size));
		numBlocks = ceil(sqrt((size + blockSizeX * blockSizeY - 1) / (blockSizeX * blockSizeY)));
		dim3 blocks8(numBlocks, numBlocks);
		cudaEventRecord(start, 0);
		addMatrixKernel << <blocks8, threads >> > (resFloatGPU, MiB8floatGPU, MiB8floatGPU, n);
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas dodawania macierzy 8MiB float na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		addMatrixCPU(resFloatCPU, MiB8floatCPU, MiB8floatCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas dodawania macierzy 8MiB float na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		mulMatrixKernel << <blocks8, threads >> > (resFloatGPU, MiB8floatGPU, MiB8floatGPU, n);
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas mno¿enia macierzy 8MiB float na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		mulMatrixCPU(resFloatCPU, MiB8floatCPU, MiB8floatCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas mno¿enia macierzy 8MiB float na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		size = 16 * 1024 * 1024 / sizeof(float);
		n = floor(sqrt(size));
		numBlocks = ceil(sqrt((size + blockSizeX * blockSizeY - 1) / (blockSizeX * blockSizeY)));
		dim3 blocks16(numBlocks, numBlocks);
		cudaEventRecord(start, 0);
		addMatrixKernel << <blocks16, threads >> > (resFloatGPU, MiB16floatGPU, MiB16floatGPU, n);
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas dodawania macierzy 16MiB float na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		addMatrixCPU(resFloatCPU, MiB16floatCPU, MiB16floatCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas dodawania macierzy 16MiB float na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		mulMatrixKernel << <blocks16, threads >> > (resFloatGPU, MiB16floatGPU, MiB16floatGPU, n);
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas mno¿enia macierzy 16MiB float na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		mulMatrixCPU(resFloatCPU, MiB16floatCPU, MiB16floatCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas mno¿enia macierzy 16MiB float na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaFree(MiB1floatGPU);
		cudaFree(MiB4floatGPU);
		cudaFree(MiB8floatGPU);
		cudaFree(MiB16floatGPU);
		cudaFree(resFloatGPU);

		free(MiB1floatCPU);
		free(MiB4floatCPU);
		free(MiB8floatCPU);
		free(MiB16floatCPU);
		free(resFloatCPU);
	}

	{
		MiB1doubleCPU = (double*)malloc(1024 * 1024);
		MiB4doubleCPU = (double*)malloc(4 * 1024 * 1024);
		MiB8doubleCPU = (double*)malloc(8 * 1024 * 1024);
		MiB16doubleCPU = (double*)malloc(16 * 1024 * 1024);
		resDoubleCPU = (double*)malloc(16 * 1024 * 1024);

		cudaMalloc(&MiB1doubleGPU, 1024 * 1024);
		cudaMalloc(&MiB4doubleGPU, 4 * 1024 * 1024);
		cudaMalloc(&MiB8doubleGPU, 8 * 1024 * 1024);
		cudaMalloc(&MiB16doubleGPU, 16 * 1024 * 1024);
		cudaMalloc(&resDoubleGPU, 16 * 1024 * 1024);

		size = 1024 * 1024 / sizeof(double);
		n = floor(sqrt(size));
		numBlocks = ceil(sqrt((size + blockSizeX * blockSizeY - 1) / (blockSizeX * blockSizeY)));
		dim3 blocks1(numBlocks, numBlocks);
		cudaEventRecord(start, 0);
		addMatrixKernel << <blocks1, threads >> > (resDoubleGPU, MiB1doubleGPU, MiB1doubleGPU, n);
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas dodawania macierzy 1MiB double na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		addMatrixCPU(resDoubleCPU, MiB1doubleCPU, MiB1doubleCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas dodawania macierzy 1MiB double na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		mulMatrixKernel << <blocks1, threads >> > (resDoubleGPU, MiB1doubleGPU, MiB1doubleGPU, n);
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas mno¿enia macierzy 1MiB double na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		mulMatrixCPU(resDoubleCPU, MiB1doubleCPU, MiB1doubleCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas mno¿enia macierzy 1MiB double na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		size = 4 * 1024 * 1024 / sizeof(double);
		n = floor(sqrt(size));
		numBlocks = ceil(sqrt((size + blockSizeX * blockSizeY - 1) / (blockSizeX * blockSizeY)));
		dim3 blocks4(numBlocks, numBlocks);
		cudaEventRecord(start, 0);
		addMatrixKernel << <blocks4, threads >> > (resDoubleGPU, MiB4doubleGPU, MiB4doubleGPU, n);
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas dodawania macierzy 4MiB double na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		addMatrixCPU(resDoubleCPU, MiB4doubleCPU, MiB4doubleCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas dodawania macierzy 4MiB double na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		mulMatrixKernel << <blocks4, threads >> > (resDoubleGPU, MiB4doubleGPU, MiB4doubleGPU, n);
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas mno¿enia macierzy 4MiB double na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		mulMatrixCPU(resDoubleCPU, MiB4doubleCPU, MiB4doubleCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas mno¿enia macierzy 4MiB double na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		size = 8 * 1024 * 1024 / sizeof(double);
		n = floor(sqrt(size));
		numBlocks = ceil(sqrt((size + blockSizeX * blockSizeY - 1) / (blockSizeX * blockSizeY)));
		dim3 blocks8(numBlocks, numBlocks);
		cudaEventRecord(start, 0);
		addMatrixKernel << <blocks8, threads >> > (resDoubleGPU, MiB8doubleGPU, MiB8doubleGPU, n);
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas dodawania macierzy 8MiB double na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		addMatrixCPU(resDoubleCPU, MiB8doubleCPU, MiB8doubleCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas dodawania macierzy 8MiB double na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		mulMatrixKernel << <blocks8, threads >> > (resDoubleGPU, MiB8doubleGPU, MiB8doubleGPU, n);
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas mno¿enia macierzy 8MiB double na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		mulMatrixCPU(resDoubleCPU, MiB8doubleCPU, MiB8doubleCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas mno¿enia macierzy 8MiB double na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		size = 16 * 1024 * 1024 / sizeof(double);
		n = floor(sqrt(size));
		numBlocks = ceil(sqrt((size + blockSizeX * blockSizeY - 1) / (blockSizeX * blockSizeY)));
		dim3 blocks16(numBlocks, numBlocks);
		cudaEventRecord(start, 0);
		addMatrixKernel << <blocks16, threads >> > (resDoubleGPU, MiB16doubleGPU, MiB16doubleGPU, n);
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas dodawania macierzy 16MiB double na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		addMatrixCPU(resDoubleCPU, MiB16doubleCPU, MiB16doubleCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas dodawania macierzy 16MiB double na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaEventRecord(start, 0);
		mulMatrixKernel << <blocks16, threads >> > (resDoubleGPU, MiB16doubleGPU, MiB16doubleGPU, n);
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas mno¿enia macierzy 16MiB double na GPU w ms: %f\n", timer);

		startOMP = omp_get_wtime();
		mulMatrixCPU(resDoubleCPU, MiB16doubleCPU, MiB16doubleCPU, size);
		stopOMP = omp_get_wtime();
		printf("Czas mno¿enia macierzy 16MiB double na CPU w ms: %f\n", 1000.0 * (stopOMP - startOMP));

		cudaFree(MiB1doubleGPU);
		cudaFree(MiB4doubleGPU);
		cudaFree(MiB8doubleGPU);
		cudaFree(MiB16doubleGPU);
		cudaFree(resDoubleGPU);

		free(MiB1doubleCPU);
		free(MiB4doubleCPU);
		free(MiB8doubleCPU);
		free(MiB16doubleCPU);
		free(resDoubleCPU);
	}
#endif
#ifdef ZADANIE5
	float *MiB1floatGPU, *MiB8floatGPU, *MiB96floatGPU, *MiB128floatGPU, *resFloatGPU;

	cudaSetDevice(0);

	cudaEvent_t start, stop;
	float timer = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int blockSizeX = 32;
	int blockSizeY = 32;
	int size;
	int n;
	int numBlocks;
	dim3 threads(blockSizeX, blockSizeY);

	{

		cudaMalloc(&MiB1floatGPU, 1024 * 1024);
		cudaMalloc(&MiB8floatGPU, 8 * 1024 * 1024);
		cudaMalloc(&MiB96floatGPU, 96 * 1024 * 1024);
		cudaMalloc(&MiB128floatGPU, 128 * 1024 * 1024);
		cudaMalloc(&resFloatGPU, 128 * 1024 * 1024);

		size = 1024 * 1024 / sizeof(float);
		n = floor(sqrt(size));
		numBlocks = ceil(sqrt((size + blockSizeX * blockSizeY - 1) / (blockSizeX * blockSizeY)));
		dim3 blocks1(numBlocks, numBlocks);

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeLinear;
		resDesc.res.linear.devPtr = MiB1floatGPU;
		resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
		resDesc.res.linear.desc.x = 32;
		resDesc.res.linear.sizeInBytes = 1024 * 1024;
		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.readMode = cudaReadModeElementType;
		cudaTextureObject_t tex = 0;
		cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

		cudaEventRecord(start, 0);
		addMatrixKernel << <blocks1, threads >> > (resFloatGPU, tex, tex, n);
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas dodawania macierzy 1MiB float na GPU w ms: %f\n", timer);

		cudaEventRecord(start, 0);
		mulMatrixKernel << <blocks1, threads >> > (resFloatGPU, tex, tex, n);
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas mno¿enia macierzy 1MiB float na GPU w ms: %f\n", timer);

		
		size = 8 * 1024 * 1024 / sizeof(float);
		n = floor(sqrt(size));
		numBlocks = ceil(sqrt((size + blockSizeX * blockSizeY - 1) / (blockSizeX * blockSizeY)));
		dim3 blocks8(numBlocks, numBlocks);

		resDesc.res.linear.devPtr = MiB8floatGPU;
		resDesc.res.linear.sizeInBytes = 8 * 1024 * 1024;
		cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

		cudaEventRecord(start, 0);
		addMatrixKernel << <blocks8, threads >> > (resFloatGPU, tex, tex, n);
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas dodawania macierzy 8MiB float na GPU w ms: %f\n", timer);

		cudaEventRecord(start, 0);
		mulMatrixKernel << <blocks8, threads >> > (resFloatGPU, tex, tex, n);
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas mno¿enia macierzy 8MiB float na GPU w ms: %f\n", timer);


		size = 96 * 1024 * 1024 / sizeof(float);
		n = floor(sqrt(size));
		numBlocks = ceil(sqrt((size + blockSizeX * blockSizeY - 1) / (blockSizeX * blockSizeY)));
		dim3 blocks96(numBlocks, numBlocks);

		resDesc.res.linear.devPtr = MiB96floatGPU;
		resDesc.res.linear.sizeInBytes = 96 * 1024 * 1024;
		cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

		cudaEventRecord(start, 0);
		addMatrixKernel << <blocks96, threads >> > (resFloatGPU, tex, tex, n);
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas dodawania macierzy 96MiB float na GPU w ms: %f\n", timer);

		cudaEventRecord(start, 0);
		mulMatrixKernel << <blocks96, threads >> > (resFloatGPU, tex, tex, n);
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas mno¿enia macierzy 96MiB float na GPU w ms: %f\n", timer);
		

		size = 128 * 1024 * 1024 / sizeof(float);
		n = floor(sqrt(size));
		numBlocks = ceil(sqrt((size + blockSizeX * blockSizeY - 1) / (blockSizeX * blockSizeY)));
		dim3 blocks128(numBlocks, numBlocks);

		resDesc.res.linear.devPtr = MiB128floatGPU;
		resDesc.res.linear.sizeInBytes = 128 * 1024 * 1024;
		cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

		cudaEventRecord(start, 0);
		addMatrixKernel << <blocks128, threads >> > (resFloatGPU, tex, tex, n);
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas dodawania macierzy 128MiB float na GPU w ms: %f\n", timer);

		cudaEventRecord(start, 0);
		mulMatrixKernel << <blocks128, threads >> > (resFloatGPU, tex, tex, n);
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&timer, start, stop);
		printf("Czas mno¿enia macierzy 128MiB float na GPU w ms: %f\n", timer);

		cudaFree(MiB1floatGPU);
		cudaFree(MiB8floatGPU);
		cudaFree(MiB96floatGPU);
		cudaFree(MiB128floatGPU);
		cudaFree(resFloatGPU);
	}
#endif


	return 0;
}

#ifdef ZADANIE3

void addCPU(int *c, int *a, int *b, int size)
{
	for (int i = 0; i < size; i++)
	{
		c[i] = a[i] + b[i];
	}
}

void mulCPU(int *c, int *a, int *b, int size)
{
	for (int i = 0; i < size; i++)
	{
		c[i] = a[i] * b[i];
	}
}

void powCPU(int *c, int *a, int *b, int size)
{
	int result;
	for (int i = 0; i < size; i++)
	{
		result = 1;
		for (int j = 0; j < b[i]; j++)
		{
			result *= a[i];
		}
		c[i] = result;
	}
}

void addCPU(float *c, float *a, float *b, int size)
{
	for (int i = 0; i < size; i++)
	{
		c[i] = a[i] + b[i];
	}
}

void mulCPU(float *c, float *a, float *b, int size)
{
	for (int i = 0; i < size; i++)
	{
		c[i] = a[i] * b[i];
	}
}

void powCPU(float *c, float *a, float *b, int size)
{
	float result;
	for (int i = 0; i < size; i++)
	{
		result = 1;
		for (int j = 0; j < b[i]; j++)
		{
			result *= a[i];
		}
		c[i] = result;
	}
}

void addCPU(double *c, double *a, double *b, int size)
{
	for (int i = 0; i < size; i++)
	{
		c[i] = a[i] + b[i];
	}
}

void mulCPU(double *c, double *a, double *b, int size)
{
	for (int i = 0; i < size; i++)
	{
		c[i] = a[i] * b[i];
	}
}

void powCPU(double *c, double *a, double *b, int size)
{
	double result;
	for (int i = 0; i < size; i++)
	{
		result = 1;
		for (int j = 0; j < b[i]; j++)
		{
			result *= a[i];
		}
		c[i] = result;
	}
}



#endif

#ifdef ZADANIE4

void addMatrixCPU(float *c, float *a, float *b, int size)
{
	int sizeRounded = floor(sqrt(size) - 1);
	if (sizeRounded * sizeRounded >= size)
	{
		printf("addMatrixCPU: B³¹d w trakcie zaokr¹glania rozmiarów macierzy.");
		return;
	}
	for (int i = 0; i < sizeRounded; i++)
	{
		for (int j = 0; j < sizeRounded; j++)
		{
			c[i * sizeRounded + j] = a[i * sizeRounded + j] + b[i * sizeRounded + j];
		}
	}
}

void mulMatrixCPU(float *c, float *a, float *b, int size)
{
	int sizeRounded = floor(sqrt(size) - 1);
	if (sizeRounded * sizeRounded >= size)
	{
		printf("mulMatrixCPU: B³¹d w trakcie zaokr¹glania rozmiarów macierzy.");
		return;
	}
	for (int i = 0; i < sizeRounded; i++)
	{
		for (int j = 0; j < sizeRounded; j++)
		{
			c[i * sizeRounded + j] = 0;
			for (int k = 0; k < sizeRounded; k++)
			{
				c[i * sizeRounded + j] += a[i * sizeRounded + k] * b[k * sizeRounded + j];
			}
		}
	}
}

void addMatrixCPU(double *c, double *a, double *b, int size)
{
	int sizeRounded = floor(sqrt(size) - 1);
	if (sizeRounded * sizeRounded >= size)
	{
		printf("addMatrixCPU: B³¹d w trakcie zaokr¹glania rozmiarów macierzy.");
		return;
	}
	for (int i = 0; i < sizeRounded; i++)
	{
		for (int j = 0; j < sizeRounded; j++)
		{
			c[i * sizeRounded + j] = a[i * sizeRounded + j] + b[i * sizeRounded + j];
		}
	}
}

void mulMatrixCPU(double *c, double *a, double *b, int size)
{
	int sizeRounded = floor(sqrt(size) - 1);
	if (sizeRounded * sizeRounded >= size)
	{
		printf("mulMatrixCPU: B³¹d w trakcie zaokr¹glania rozmiarów macierzy.");
		return;
	}
	for (int i = 0; i < sizeRounded; i++)
	{
		for (int j = 0; j < sizeRounded; j++)
		{
			c[i * sizeRounded + j] = 0;
			for (int k = 0; k < sizeRounded; k++)
			{
				c[i * sizeRounded + j] += a[i * sizeRounded + k] * b[k * sizeRounded + j];
			}
		}
	}
}

#endif