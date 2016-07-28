
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#include "CL/cl2.hpp"

#include "Stream.h"

#define IMPLEMENTATION_STRING "OpenCL"

template <class T>
class OCLStream : public Stream<T>
{
  protected:
    // Size of arrays
    unsigned int array_size;
    unsigned int image_size;

    // Device side pointers to arrays
    cl::Image2D d_a;
    cl::Image2D d_b;
    cl::Image2D d_c;

    // OpenCL objects
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;

    cl::KernelFunctor<cl::Image2D, cl::Image2D> *copy_kernel;
    cl::KernelFunctor<cl::Image2D, cl::Image2D> * mul_kernel;
    cl::KernelFunctor<cl::Image2D, cl::Image2D, cl::Image2D> *add_kernel;
    cl::KernelFunctor<cl::Image2D, cl::Image2D, cl::Image2D> *triad_kernel;

  public:

    OCLStream(const unsigned int, const int);
    ~OCLStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;

    virtual void write_arrays(const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& c) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;

};

// Populate the devices list
void getDeviceList(void);
