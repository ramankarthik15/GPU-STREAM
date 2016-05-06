
#include "OMP40Stream.h"

template <class T>
OMP40Stream<T>::OMP40Stream(const unsigned int ARRAY_SIZE)
{
  array_size = ARRAY_SIZE;
  omp_a = (double *)malloc(sizeof(double)*array_size);
  omp_b = (double *)malloc(sizeof(double)*array_size);
  omp_c = (double *)malloc(sizeof(double)*array_size);
}

template <class T>
OMP40Stream<T>::~OMP40Stream()
{
}


template <class T>
void OMP40Stream<T>::write_arrays(const std::vector<T>& h_a, const std::vector<T>& h_b, const std::vector<T>& h_c)
{
}

template <class T>
void OMP40Stream<T>::read_arrays(std::vector<T>& h_a, std::vector<T>& h_b, std::vector<T>& h_c)
{
}

template <class T>
void OMP40Stream<T>::copy()
{
  #pragma omp target teams distribute parallel for simd
  for (int i = 0; i < ARRAY_SIZE; i++)
  {
    omp_c[i] = omp_a[i];
  }
}

template <class T>
void OMP40Stream<T>::mul()
{
  const T scalar = 3.0;
  #pragma omp target teams distribute parallel for simd
  for (int i = 0; i < ARRAY_SIZE; i++)
  {
    omp_b[i] = scalar * omp_c[i];
  }
}

template <class T>
void OMP40Stream<T>::add()
{
  #pragma omp target teams distribute parallel for simd
  for (int i = 0; i < ARRAY_SIZE; i++)
  {
    omp_c[i] = omp_a[i] + omp_b[i];
  }
}

template <class T>
void OMP40Stream<T>::triad()
{
  const T scalar = 3.0;
  #pragma omp target teams distribute parallel for simd
  for (int i = 0; i < ARRAY_SIZE; i++)
  {
    omp_a[i] = omp_b[i] + scalar * omp_c[i];
  }
}

void listDevices(void)
{
  std::cout << "0: CPU" << std::endl;
}

std::string getDeviceName(const int)
{
  return std::string("Device name unavailable");
}

std::string getDeviceDriver(const int)
{
  return std::string("Device driver unavailable");
}


template class OMP40Stream<double>;

