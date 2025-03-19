#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>

using tt_i1 = int8_t;
using tt_i8 = int8_t;
using tt_i16 = int16_t;
using tt_i32 = int32_t;
using tt_i64 = int64_t;
using tt_u8 = uint8_t;
using tt_u16 = uint16_t;
using tt_u32 = uint32_t;
using tt_u64 = uint64_t;
using tt_f32 = float;
using tt_fp32 = float;
using tt_fp64 = double;

#pragma pack(push, 1)
template <unsigned M /* Mantissa bits */, unsigned E /* Exponent bits */,
          unsigned B /* Bias */ = (1u << (E - 1)) - 1,
          bool NI /* Convert NaN or Inf */ = B != 127 && B != (1u << E) - 1,
          bool DN /* Convert denormalized */ = B != 127>
struct tt_fpme {
  uint32_t mantissa : M;
  uint32_t exponent : E;
  uint32_t sign : 1;

  constexpr static unsigned mantissaBitWidth() { return M; }

  constexpr static unsigned exponentBitWidth() { return E; }

  explicit operator float() const {
    constexpr unsigned mShift = 23 - M;
    constexpr uint32_t eAdjust = 127 - B;

    if constexpr (NI) {
      constexpr uint32_t maxExp = (1u << E) - 1;
      if (exponent == maxExp) { // NaN or Inf
        uint32_t v = (sign << 31) | 0x7F800000 | (mantissa << mShift);
        return *reinterpret_cast<float *>(&v);
      }
    }

    if constexpr (DN) {
      if (!exponent) {
        // Denormalized value = (mantissa / 2^mantissa_bits) * 2^(1 - bias)
        static const float mul = std::pow(2.0f, 1 - static_cast<int>(B)) /
                                 static_cast<float>(1u << M);
        float v = static_cast<float>(mantissa) * mul;
        return sign ? -v : v;
      }
    }

    uint32_t v =
        (sign << 31) | ((exponent + eAdjust) << 23) | (mantissa << mShift);
    return *reinterpret_cast<float *>(&v);
  }

  bool operator==(const tt_fpme &other) const {
    return (mantissa == other.mantissa) && (exponent == other.exponent) &&
           (sign == other.sign);
  }

  bool operator!=(const tt_fpme &other) const { return !(*this == other); }

  friend std::ostream &operator<<(std::ostream &os, const tt_fpme &value) {
    return os << static_cast<float>(value)
              << " (m=" << std::bitset<M>(value.mantissa)
              << ", e=" << std::bitset<E>(value.exponent)
              << ", s=" << value.sign << ')';
  }
};
#pragma pack(pop)

template <typename T>
bool isClose(T x, T y, T rtol = 1e-05, T atol = 1e-08, bool nanEq = false) {
  if (x == y) {
    return true;
  }
  if (std::isnan(x) || std::isnan(y)) {
    return nanEq && std::isnan(x) && std::isnan(y);
  }
  return std::abs(x - y) <= atol + rtol * std::abs(y);
}

#define TT_FPB(N, M, E, B)                                                     \
  struct tt_##N : tt_fpme<M, E, B> {};                                         \
                                                                               \
  bool isClose(tt_##N x, tt_##N y, float rtol = 1e-05, float atol = 1e-08,     \
               bool nanEq = false) {                                           \
    return isClose(static_cast<float>(x), static_cast<float>(y), rtol, atol,   \
                   nanEq);                                                     \
  }                                                                            \
                                                                               \
  namespace std {                                                              \
  std::string to_string(const tt_##N &value) {                                 \
    std::ostringstream oss;                                                    \
    oss << value;                                                              \
    return oss.str();                                                          \
  }                                                                            \
  }
#define TT_FP(N, M, E) TT_FPB(N, M, E, ((1u << (E - 1)) - 1))

TT_FP(fp8e5, 2, 5)
TT_FP(fp8e4nv, 3, 4)
TT_FPB(fp8e4b15, 3, 4, 15)
TT_FP(fp16, 10, 5)
TT_FP(bf16, 7, 8)

sycl::device findDevice(const std::string &name) {
  for (const auto &platform : sycl::platform::get_platforms()) {
    for (const auto &dev : platform.get_devices()) {
      if ((name.empty() &&
           dev.get_backend() == sycl::backend::ext_oneapi_level_zero) ||
          (!name.empty() && dev.get_info<sycl::info::device::name>().find(
                                name) != std::string::npos)) {
        std::cout << "Using device '"
                  << dev.get_info<sycl::info::device::name>() << "'."
                  << std::endl;
        return dev;
      }
    }
  }
  throw std::runtime_error("Device not found");
}

template <typename T = uint8_t>
static std::vector<T> readFile(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  const auto size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<T> binary(size / sizeof(T));
  file.read(reinterpret_cast<char *>(binary.data()), size);
  return binary;
}

template <typename T = uint8_t> struct MemRef {
  T *ptr;

  MemRef(sycl::queue &queue, size_t size)
      : MemRef(queue, sycl::malloc_device<T>(size, queue)) {
    if (!ptr) {
      throw std::runtime_error("Failed to allocate memory");
    }
  }
  MemRef(sycl::queue &queue, T *ptr) : ptr(ptr), queue(queue) {}
  ~MemRef() { sycl::free(ptr, queue); }

private:
  sycl::queue &queue;
};

struct Launcher {

  explicit Launcher(sycl::queue &queue, const std::string &path,
                    const std::string &name, bool isNative)
      : queue(queue),
        kernel(createKernel(
            queue,
            createL0Kernel(createL0Module(queue, path, isNative), name))) {}

  template <typename T = uint8_t> MemRef<T> copyFrom(std::vector<T> &data) {
    MemRef<T> mem(queue, data.size());
    queue.memcpy(mem.ptr, data.data(), data.size() * sizeof(T));
    return mem;
  }

  template <typename T = uint8_t>
  void copyTo(MemRef<T> &src, std::vector<T> &dst) {
    queue.memcpy(dst.data(), src.ptr, dst.size() * sizeof(T));
  }

  sycl::event submit(size_t gridX, size_t gridY, size_t gridZ, size_t numWraps,
                     size_t threadsPerWrap,
                     const std::function<void(sycl::handler &)> &init) const {
    return queue.submit([&](sycl::handler &h) {
      init(h);
      sycl::range local(1, 1, numWraps * threadsPerWrap);
      sycl::range global(gridZ, gridY, gridX * local[2]);
      h.parallel_for(sycl::nd_range{global, local}, kernel);
    });
  }

private:
  sycl::queue &queue;
  sycl::kernel kernel;

  static ze_module_handle_t createL0Module(const sycl::queue &queue,
                                           const std::string &path,
                                           bool isNative) {
    auto moduleBin = readFile(path);
    ze_module_desc_t moduleDsc = {};
    moduleDsc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    moduleDsc.format =
        isNative ? ZE_MODULE_FORMAT_NATIVE : ZE_MODULE_FORMAT_IL_SPIRV;
    moduleDsc.inputSize = moduleBin.size();
    moduleDsc.pInputModule = moduleBin.data();
    ze_module_build_log_handle_t buildLog;
    ze_module_handle_t module;
    ze_context_handle_t l0Ctx =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
            queue.get_context());
    ze_device_handle_t l0Dev =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
            queue.get_device());
    if (zeModuleCreate(l0Ctx, l0Dev, &moduleDsc, &module, &buildLog) !=
        ZE_RESULT_SUCCESS) {
      size_t logLen = 0;
      zeModuleBuildLogGetString(buildLog, &logLen, nullptr);
      const auto msg = "L0 module build failed:\n";
      const auto msgLen = std::strlen(msg);
      std::string err(logLen + msgLen, '\0');
      err.append(msg);
      zeModuleBuildLogGetString(buildLog, &logLen, err.data() + msgLen);
      zeModuleBuildLogDestroy(buildLog);
      throw std::runtime_error(err);
    }
    return module;
  }

  static std::pair<ze_module_handle_t, ze_kernel_handle_t>
  createL0Kernel(const ze_module_handle_t module, const std::string &name) {
    ze_kernel_handle_t kernel;
    ze_kernel_desc_t kernelDsc = {};
    kernelDsc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    kernelDsc.pKernelName = name.data();
    if (zeKernelCreate(module, &kernelDsc, &kernel) != ZE_RESULT_SUCCESS) {
      throw std::runtime_error("Failed to create kernel");
    }
    return {module, kernel};
  }

  static sycl::kernel
  createKernel(const sycl::queue &queue,
               const std::pair<ze_module_handle_t, ze_kernel_handle_t> &pair) {
    const auto ctx =
        queue.get_device().get_platform().ext_oneapi_get_default_context();
    const auto bundle =
        sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero,
                                 sycl::bundle_state::executable>(
            {pair.first, sycl::ext::oneapi::level_zero::ownership::transfer},
            ctx);
    return sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
        {bundle, pair.second,
         sycl::ext::oneapi::level_zero::ownership::transfer},
        ctx);
  }
};
