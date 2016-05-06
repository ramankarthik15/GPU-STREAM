
#pragma once

#include <iostream>
#include <stdexcept>

#include "Stream.h"

#define IMPLEMENTATION_STRING "OpenMP 4.0"

extern double *omp_a;
extern double *omp_b;
extern double *omp_c;
extern unsigned int ARRAY_SIZE;

template <class T>
class OMP40Stream : public Stream<T>
{
  protected:
    // Size of arrays
    unsigned int array_size;

  public:
    OMP40Stream(const unsigned int);
    ~OMP40Stream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;

    virtual void write_arrays(const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& c) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;

};

