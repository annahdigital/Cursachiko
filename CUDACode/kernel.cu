#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cmath>
#include <chrono>
#include <cufft.h>
#include <cufftXt.h>
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::milli;

using namespace std;

//____________________ NOTES__________________________________
// ___________________________________________________________
// for our pc cache size is 6 MB
// GPU is 256 MB
// => dataset sizes are (cause we use cufftComplex(8) arrays):
// + cache:  max 1572864 / 4 for each vector
// + GPU:  max 67108864 / 4 for each vector

int N = 67108864 / 4;

// Data padding for fitting in after calculations
void PadData(const cufftComplex*, cufftComplex**, int);

// Normalization for number after multiplipaction
void Normalize(cufftComplex*, cufftComplex**, int);

// Complex multiplication
static __device__ __host__ inline cufftComplex ComplexMul(cufftComplex, cufftComplex);
// Complex pointwise multiplication
static __global__ void ComplexPointwiseMul(cufftComplex*, const cufftComplex*, int);


int main(int argc, char* argv[]) {

	cufftComplex* vec1_padded, * vec1, * vec2, * vec2_padded;
	vec1 = (cufftComplex*)malloc(N * sizeof(cufftComplex));
	vec2 = (cufftComplex*)malloc(N * sizeof(cufftComplex));

	// initialization
	for (int i = 0; i < N; i++)
	{
		vec1[i].x = 1;
		vec1[i].y = 0;
		vec2[i].x = 1;
		vec2[i].y = 0;
	}

	auto startTime = high_resolution_clock::now();
	// padding data
	size_t new_size = 1;
	while (new_size < N)
		new_size <<= 1;
	new_size <<= 1;
	PadData(vec1, &vec1_padded, new_size);
	PadData(vec2, &vec2_padded, new_size);

	cudaFree(vec1);
	cudaFree(vec2);

	// save arrays from host to device
	cufftComplex* vec1_DEVICE;
	cudaMalloc((void**)&vec1_DEVICE, new_size * sizeof(cufftComplex));
	cudaMemcpy(vec1_DEVICE, vec1_padded, new_size * sizeof(cufftComplex), cudaMemcpyHostToDevice);

	cufftComplex* vec2_DEVICE;
	cudaMalloc((void**)&vec2_DEVICE, new_size * sizeof(cufftComplex));
	cudaMemcpy(vec2_DEVICE, vec2_padded, new_size * sizeof(cufftComplex), cudaMemcpyHostToDevice);

	cudaFree(vec1_padded);
	cudaFree(vec2_padded);

	cufftHandle plan_VEC1;
	cufftPlan1d(&plan_VEC1, new_size, CUFFT_C2C, 1);
	cufftExecC2C(plan_VEC1, vec1_DEVICE, vec1_DEVICE, CUFFT_FORWARD);

	cufftHandle plan_VEC2;
	cufftPlan1d(&plan_VEC2, new_size, CUFFT_C2C, 1);
	cufftExecC2C(plan_VEC2, vec2_DEVICE, vec2_DEVICE, CUFFT_FORWARD);

	// Multiplication
	ComplexPointwiseMul << <32, 256 >> > (vec1_DEVICE, vec2_DEVICE, new_size * sizeof(cufftComplex));
	cufftExecC2C(plan_VEC1, vec1_DEVICE, vec1_DEVICE, CUFFT_INVERSE);

	cufftComplex* test_out;
	test_out = (cufftComplex*)malloc(new_size * sizeof(cufftComplex));
	cudaMemcpy(test_out, vec1_DEVICE, new_size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	cufftComplex* test_out2;
	Normalize(test_out, &test_out2, new_size);


	/*for (int i = 0; i < new_size; i++)
	{
		cout << test_out2[i].x;
		cout << endl;
	}
	cout << endl << "____________" << endl;*/


	auto endTime = high_resolution_clock::now();
	auto overallTime = duration_cast<duration<double, milli>>(endTime - startTime).count();
	cout << overallTime << " ms" << '\n';

	cudaFree(vec1_DEVICE);
	cudaFree(vec2_DEVICE);
	cudaFree(test_out);
	cudaFree(test_out2);
	cufftDestroy(plan_VEC1);
	cufftDestroy(plan_VEC2);
}

// Complex multiplication
static __device__ __host__ inline cufftComplex ComplexMul(cufftComplex a, cufftComplex b)
{
	cufftComplex c;
	c.x = a.x * b.x - a.y * b.y;
	c.y = a.x * b.y + a.y * b.x;
	return c;
}


// Complex pointwise multiplication
static __global__ void ComplexPointwiseMul(cufftComplex* a, const cufftComplex* b, int size)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = threadID; i < size; i += numThreads)
		a[i] = ComplexMul(a[i], b[i]);
}


void PadData(const cufftComplex* signal, cufftComplex** padded_signal, int new_size) {
	cufftComplex* new_data = reinterpret_cast<cufftComplex*>(malloc(sizeof(cufftComplex) * new_size));
	memcpy(new_data + 0, signal, N * sizeof(cufftComplex));
	memset(new_data + N, 0, (new_size - N) * sizeof(cufftComplex));
	*padded_signal = new_data;
}

void Normalize(cufftComplex* signal, cufftComplex** normalized_signal, int new_size) {
	cufftComplex* new_data = reinterpret_cast<cufftComplex*>(malloc(sizeof(cufftComplex) * new_size));
	int zeros = 0;
	for (int i = new_size - 1; i >= 0; i--)
		if (round(signal[i].x) == 0) zeros++;
		else break;
	memcpy(new_data + zeros, signal, (new_size - zeros) * sizeof(cufftComplex));
	memset(new_data + 0, 0, zeros * sizeof(cufftComplex));
	long long carry = 0;
	for (size_t i = 0; i < new_size; ++i) {
		new_data[i].x = round(new_data[i].x / new_size);
		new_data[i].x += carry;
		carry = new_data[i].x / 10;
		new_data[i].x = (int)(new_data[i].x) % 10;
	}
	*normalized_signal = new_data;
}