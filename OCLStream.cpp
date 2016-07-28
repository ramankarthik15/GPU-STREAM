
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "OCLStream.h"

#include <cmath>

// Cache list of devices
bool cached = false;
std::vector<cl::Device> devices;
void getDeviceList(void);

std::string kernels{R"CLC(

  constant TYPE scalar = 0.3;

  kernel void copy(
    read_only  image2d_t a,
    write_only image2d_t c)
  {
    const size_t x = get_global_id(0);
    const size_t y = get_global_id(1);
    const TYPE _a = read_imagef(a, (int2)(x,y)).x;
    write_imagef(c, (int2)(x,y), _a);
  }

  kernel void mul(
    write_only image2d_t b,
    read_only  image2d_t c)
  {
    const size_t x = get_global_id(0);
    const size_t y = get_global_id(1);
    const TYPE _c = read_imagef(c, (int2)(x,y)).x;
    write_imagef(b, (int2)(x,y), scalar * _c);
  }

  kernel void add(
    read_only  image2d_t a,
    read_only  image2d_t b,
    write_only image2d_t c)
  {
    const size_t x = get_global_id(0);
    const size_t y = get_global_id(1);
    const TYPE _a = read_imagef(a, (int2)(x,y)).x;
    const TYPE _b = read_imagef(b, (int2)(x,y)).x;
    write_imagef(c, (int2)(x,y), _a + _b);
  }

  kernel void triad(
    write_only image2d_t a,
    read_only  image2d_t b,
    read_only  image2d_t c)
  {
    const size_t x = get_global_id(0);
    const size_t y = get_global_id(1);
    const TYPE _b = read_imagef(b, (int2)(x,y)).x;
    const TYPE _c = read_imagef(c, (int2)(x,y)).x;
    write_imagef(a, (int2)(x,y), _b + scalar*_c);
  }

)CLC"};


template <class T>
OCLStream<T>::OCLStream(const unsigned int ARRAY_SIZE, const int device_index)
{
  if (!cached)
    getDeviceList();

  // Setup default OpenCL GPU
  if (device_index >= devices.size())
    throw std::runtime_error("Invalid device index");
  device = devices[device_index];

  // Print out device information
  std::cout << "Using OpenCL device " << getDeviceName(device_index) << std::endl;
  std::cout << "Driver: " << getDeviceDriver(device_index) << std::endl;

  context = cl::Context(device);
  queue = cl::CommandQueue(context);

  // Create program
  cl::Program program(context, kernels);
  try
  {
    if (sizeof(T) == sizeof(double))
    {
      // Check device can do double
      if (!device.getInfo<CL_DEVICE_DOUBLE_FP_CONFIG>())
        throw std::runtime_error("Device does not support double precision, please use --float");
      program.build("-DTYPE=double");
    }
    else if (sizeof(T) == sizeof(float))
      program.build("-DTYPE=float");
    }
  catch (cl::BuildError& err)
  {
    std::cerr << err.getBuildLog()[0].second << std::endl;
  }

  // Create kernels
  copy_kernel = new cl::KernelFunctor<cl::Image2D, cl::Image2D>(program, "copy");
  mul_kernel = new cl::KernelFunctor<cl::Image2D, cl::Image2D>(program, "mul");
  add_kernel = new cl::KernelFunctor<cl::Image2D, cl::Image2D, cl::Image2D>(program, "add");
  triad_kernel = new cl::KernelFunctor<cl::Image2D, cl::Image2D, cl::Image2D>(program, "triad");

  array_size = ARRAY_SIZE;
  image_size = sqrt(ARRAY_SIZE);
  if (image_size*image_size != array_size)
  {
    throw std::runtime_error("array_size must be square for 2D images");
  }

  // Check buffers fit on the device
  cl_ulong totalmem = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
  cl_ulong maxbuffer = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
  if (maxbuffer < sizeof(T)*ARRAY_SIZE)
    throw std::runtime_error("Device cannot allocate a buffer big enough");
  if (totalmem < 3*sizeof(T)*ARRAY_SIZE)
    throw std::runtime_error("Device does not have enough memory for all 3 buffers");

  // Create images
  cl::ImageFormat format = {CL_R, CL_FLOAT};
  d_a = cl::Image2D(context, CL_MEM_READ_WRITE, format, image_size, image_size);
  d_b = cl::Image2D(context, CL_MEM_READ_WRITE, format, image_size, image_size);
  d_c = cl::Image2D(context, CL_MEM_READ_WRITE, format, image_size, image_size);
}

template <class T>
OCLStream<T>::~OCLStream()
{
  delete copy_kernel;
  delete mul_kernel;
  delete add_kernel;
  delete triad_kernel;
}

template <class T>
void OCLStream<T>::copy()
{
  (*copy_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(image_size, image_size)),
    d_a, d_c
  );
  queue.finish();
}

template <class T>
void OCLStream<T>::mul()
{
  (*mul_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(image_size, image_size)),
    d_b, d_c
  );
  queue.finish();
}

template <class T>
void OCLStream<T>::add()
{
  (*add_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(image_size, image_size)),
    d_a, d_b, d_c
  );
  queue.finish();
}

template <class T>
void OCLStream<T>::triad()
{
  (*triad_kernel)(
    cl::EnqueueArgs(queue, cl::NDRange(image_size, image_size)),
    d_a, d_b, d_c
  );
  queue.finish();
}

template <class T>
void OCLStream<T>::write_arrays(const std::vector<T>& a, const std::vector<T>& b, const std::vector<T>& c)
{
  cl::array<cl::size_type, 3> origin;
  origin[0] = origin[1] = origin[2] = 0;
  cl::array<cl::size_type, 3> region;
  region[0] = region[1] = image_size;
  region[2] = 1;
  queue.enqueueWriteImage(d_a, CL_TRUE, origin, region, 0, 0, a.data());
  queue.enqueueWriteImage(d_b, CL_TRUE, origin, region, 0, 0, b.data());
  queue.enqueueWriteImage(d_c, CL_TRUE, origin, region, 0, 0, c.data());
}

template <class T>
void OCLStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  cl::array<cl::size_type, 3> origin;
  origin[0] = origin[1] = origin[2] = 0;
  cl::array<cl::size_type, 3> region;
  region[0] = region[1] = image_size;
  region[2] = 1;
  queue.enqueueReadImage(d_a, CL_TRUE, origin, region, 0, 0, a.data());
  queue.enqueueReadImage(d_b, CL_TRUE, origin, region, 0, 0, b.data());
  queue.enqueueReadImage(d_c, CL_TRUE, origin, region, 0, 0, c.data());
}

void getDeviceList(void)
{
  // Get list of platforms
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  // Enumerate devices
  for (unsigned i = 0; i < platforms.size(); i++)
  {
    std::vector<cl::Device> plat_devices;
    platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &plat_devices);
    devices.insert(devices.end(), plat_devices.begin(), plat_devices.end());
  }
  cached = true;
}

void listDevices(void)
{
  getDeviceList();

  // Print device names
  if (devices.size() == 0)
  {
    std::cerr << "No devices found." << std::endl;
  }
  else
  {
    std::cout << std::endl;
    std::cout << "Devices:" << std::endl;
    for (int i = 0; i < devices.size(); i++)
    {
      std::cout << i << ": " << getDeviceName(i) << std::endl;
    }
    std::cout << std::endl;
  }


}

std::string getDeviceName(const int device)
{
  if (!cached)
    getDeviceList();

  std::string name;
  cl_device_info info = CL_DEVICE_NAME;

  if (device < devices.size())
  {
    devices[device].getInfo(info, &name);
  }
  else
  {
    throw std::runtime_error("Error asking for name for non-existant device");
  }

  return name;

}

std::string getDeviceDriver(const int device)
{
  if (!cached)
    getDeviceList();

  std::string driver;

  if (device < devices.size())
  {
    devices[device].getInfo(CL_DRIVER_VERSION, &driver);
  }
  else
  {
    throw std::runtime_error("Error asking for driver for non-existant device");
  }

  return driver;
}


template class OCLStream<float>;
template class OCLStream<double>;
