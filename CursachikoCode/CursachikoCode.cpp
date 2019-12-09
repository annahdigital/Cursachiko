#include <iostream>
#include <algorithm>
#include <vector>
#include <set>
#include <map>
#include <cmath>
#include <queue>
#include <fstream>
#include <complex>
#include <math.h>
#include <stdlib.h>
#include <numeric>
#include <iomanip>
#include <stddef.h>
#include <stdio.h>
#include <chrono>
#include <random>
#include <ratio>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>
using std::max;
using std::cout;
using std::swap;
using std::sort;
using std::complex;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::milli;
using std::random_device;
using std::sort;
using std::vector;

constexpr double pi = 3.14159265358979323846;
const complex<double> i(0, 1);
typedef complex<double> w_type;
vector<int> rev;

void fft(vector<w_type>& vec, bool inverse) {
	size_t n = vec.size();
	if (n == 1)
		return;
	// recursive division of the polynomial : even and odd coefficients
	vector<w_type> vec0(n / 2), vec1(n / 2);
	for (int i = 0, j = 0; i < n; i += 2, ++j) {
		vec0[j] = vec[i];
		vec1[j] = vec[i + 1];
	}

	fft(vec0, inverse);
	fft(vec1, inverse);

	double arg = 2 * pi / n;
	if (inverse)
		arg *= -1;
	w_type w(1), wn = std::exp(i * arg);
	// butterfly operation
	for (int i = 0; i < n / 2; ++i) {
		w_type sec_part = w * vec1[i];
		vec[i] = vec0[i] + sec_part;
		vec[i + n / 2] = vec0[i] - sec_part;
		if (inverse) {
			vec[i] /= 2;
			vec[i + n / 2] /= 2;            // division results /n because of recursion
		}
		w *= wn;
	}
}

void fft_optimized(vector<w_type>& vec, bool inverse) {
	size_t n = vec.size();

	for (int i = 0; i < n; ++i)
		if (i < rev[i])
			swap(vec[i], vec[rev[i]]);


	for (int k = 2; k <= n; k <<= 1) {
		double arg = 2 * pi / k;
		if (inverse) arg *= -1;
		w_type wn = std::exp(i * arg);
		for (int i = 0; i < n; i += k) {
			w_type w(1);
			for (int j = 0; j < k / 2; ++j) {
				w_type u = vec[i + j];
				w_type v = vec[i + j + k / 2] * w;
				vec[i + j] = u + v;
				vec[i + j + k / 2] = u - v;
				w *= wn;
			}
		}
	}

	if (inverse)
		for (int i = 0; i < n; ++i)
			vec[i] /= n;
}

void fft_openMP(vector<w_type>& vec, bool inverse) {
	size_t n = vec.size();

#pragma omp parallel for
	for (int i = 0; i < n; ++i)
		if (i < rev[i])
			swap(vec[i], vec[rev[i]]);

#pragma omp parallel
	for (int k = 2; k <= n; k <<= 1) {
		double arg = 2 * pi / k;
		if (inverse)
			arg *= -1;
		w_type wn = std::exp(i * arg);
#pragma omp for
		for (int i = 0; i < n; i += k) {
			w_type w(1);
			for (int j = 0; j < k / 2; ++j) {
				w_type u = vec[i + j];
				w_type v = vec[i + j + k / 2] * w;
				vec[i + j] = u + v;
				vec[i + j + k / 2] = u - v;
				w *= wn;
			}
		}
	}
	
	if (inverse == true) 
#pragma omp parallel for
	for (int i = 0; i < n; ++i)
		vec[i] /= n;
}


void calcRev(int n) {
	rev = vector<int>(n);
	int s = 0;
	// counting meaningful bits in n
	while ((1 << s) < n)  ++s;
	//finding position that's bit note is bit note i in reverse order
	for (int i = 0; i < n; i++)
		for (int j = 0; j < s; j++)
			if ((i & (1 << j)) != 0)
				rev[i] += (1 << (s - j - 1));
}

void calcRevOpenMP(int n) {
	int s = 0;
	rev = vector<int>(n);
	// counting meaningful bits in n
	while ((1 << s) < n)  ++s;
	//finding position that's bit note is bit note i in reverse order
#pragma parallel for
	for (int i = 0; i < n; i++)
		for (int j = 0; j < s; j++)
			if ((i & (1 << j)) != 0)
				rev[i] += (1 << (s - j - 1));
}


void polynomial_multiplication(const vector<int>& a, const std::vector<int>& b, vector<int>& result, void(*fft)(vector<w_type>&, bool),
	void(*calcRev)(int)) {
	vector<w_type> complex_a(a.begin(), a.end());
	vector<w_type> complex_b(a.begin(), a.end());
	size_t n = 1;
	while (n < max(a.size(), b.size()))
		n <<= 1;
	n <<= 1;
	complex_a.resize(n);
	complex_b.resize(n);
	result.resize(n);
	calcRev(n);

	// FFT for A and B
	fft(complex_a, false);
	fft(complex_b, false);

	// getting complex result
	for (size_t i = 0; i < n; ++i)
		complex_a[i] *= complex_b[i];
	// inverse fft
	fft(complex_a, true);

	for (size_t i = 0; i < n; ++i)
		result[i] = (int)(complex_a[i].real() + 0.5);

	// normalization of the numbers
	long long carry = 0;
	for (size_t i = 0; i < n; ++i) {
		result[i] += carry;
		carry = result[i] / 10;
		result[i] %= 10;
	}
}

void polynomial_multiplication(const vector<int>& a, const std::vector<int>& b, vector<int>& result, void(*fft)(vector<w_type>&, bool)) {
	vector<w_type> complex_a(a.begin(), a.end());
	vector<w_type> complex_b(a.begin(), a.end());
	size_t n = 1;
	while (n < max(a.size(), b.size()))
		n <<= 1;
	n <<= 1;
	complex_a.resize(n);
	complex_b.resize(n);
	result.resize(n);
	// FFT for A and B
	fft(complex_a, false);
	fft(complex_b, false);

	// getting complex result
	for (size_t i = 0; i < n; ++i)
		complex_a[i] *= complex_b[i];
	// inverse fft
	fft(complex_a, true);
	for (size_t i = 0; i < n; ++i)
		result[i] = (int)(complex_a[i].real() + 0.5);

	// normalization of the numbers
	long long carry = 0;
	for (size_t i = 0; i < n; ++i) {
		result[i] += carry;
		carry = result[i] / 10;
		result[i] %= 10;
	}
}

//					NOTES
// _______________________________________________
// for our pc cache size is 6 MB
// GPU is 256 MB
// => dataset sizes are (cause we use int arrays):
// + cache:  max 1572864 / 2 for each vector
// + GPU:  max 67108864 / 2 for each vector


int main() {

	int n = 10000000;
	//int n = 67108864;

	std::vector<int> a(n);
	cout << sizeof(a);
	std::vector<int> b;
	fill(a.begin(), a.end(), 1);
	b = a;

	vector<int> res, res0, res1;

	auto startTime = high_resolution_clock::now();
	polynomial_multiplication(a, b, res, fft);
	auto endTime = high_resolution_clock::now();
	auto overallTime23 = duration_cast<duration<double, milli>>(endTime - startTime).count();
	cout << overallTime23 << " ms" << '\n';

	startTime = high_resolution_clock::now();
	polynomial_multiplication(a, b, res0, fft_optimized, calcRev);
	endTime = high_resolution_clock::now();
	auto overallTime = duration_cast<duration<double, milli>>(endTime - startTime).count();
	cout << overallTime << " ms" << '\n';

	startTime = high_resolution_clock::now();
	polynomial_multiplication(a, b, res1, fft_openMP, calcRevOpenMP);
	endTime = high_resolution_clock::now();
	auto overallTime2 = duration_cast<duration<double, milli>>(endTime - startTime).count();
	cout << overallTime2 << " ms" << '\n';

	double diff = overallTime / overallTime2;
	cout << diff << '\n';


	/*cout << std::endl;
	for(int i = res.size() - 1; i >= 0; i--)
		cout << res[i];
	cout << '\n';
	for (int i = res0.size() - 1; i >= 0; i--)
		cout << res0[i];
	cout << '\n';
	for (int i = res1.size() - 1; i >= 0; i--)
		cout << res1[i];*/

}
