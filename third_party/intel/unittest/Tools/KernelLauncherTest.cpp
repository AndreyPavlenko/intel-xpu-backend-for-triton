#include <bitset>
#include <limits>

#include <gtest/gtest.h>

#include "third_party/intel/backend/reproducer/KernelLauncher.cpp"

template <typename T, unsigned M = T::mantissaBitWidth(),
          unsigned E = T::exponentBitWidth()>
struct FpTest {
  static_assert(sizeof(T) == (M + E + 8) / 8);
  uint32_t mantissa : M;
  uint32_t exponent : E;
  uint32_t sign : 1;
  float expected;

  static void test(const char *typeName,
                   std::initializer_list<FpTest<T>> testCases) {
    for (const auto &[mantissa, exponent, sign, expected] : testCases) {
      T value{mantissa, exponent, sign};
      const float result = static_cast<float>(value);
      ASSERT_TRUE(isClose(result, expected))
          << "Conversion from " << typeName << " to float failed! "
          << "Mantissa: " << std::bitset<M>(mantissa).to_string()
          << ". Exponent: " << std::bitset<E>(exponent).to_string()
          << ". Sign: " << sign << ", Expected: " << expected
          << ". Actual: " << result << '.';
    }
  }
};

#define FP_TEST(T, ...)                                                        \
  TEST(KernelLauncherTest, test_##T) { FpTest<T>::test(#T, __VA_ARGS__); }

FP_TEST(tt_fp16,
        {
            {0b0000000000, 0b00000, 0, 0.0f},
            {0b0000000000, 0b00000, 1, -0.0f},
            // smallest positive subnormal number
            {0b0000000001, 0b00000, 0, 0.000000059604645},
            // largest subnormal number
            {0b1111111111, 0b00000, 0, 0.000060975552f},
            // smallest positive normal number
            {0b0000000000, 0b00001, 0, 0.00006103515625f},
            // nearest value to 1/3
            {0b0101010101, 0b01101, 0, 0.33325195f},
            // largest number less than one
            {0b1111111111, 0b01110, 0, 0.99951172f},
            // one
            {0b0000000000, 0b01111, 0, 1.f},
            // smallest number larger than one
            {0b0000000001, 0b01111, 0, 1.00097656f},
            // largest normal number
            {0b1111111111, 0b11110, 0, 65504.f},
            // infinity
            {0b0000000000, 0b11111, 0, std::numeric_limits<float>::infinity()},
            // negative infinity
            {0b0000000000, 0b11111, 1, -std::numeric_limits<float>::infinity()},
            {0b0000000000, 0b10000, 1, -2},
            {0b0000000010, 0b00011, 0, 0.00024462890625},
        })

FP_TEST(tt_bf16,
        {
            // // zero
            {0b0000000, 0b00000000, 0, 0.0f},
            // // negative zero
            {0b0000000, 0b00000000, 1, -0.0f},
            // smallest positive normal number
            {0b0000000, 0b00000001, 0, 1.17549e-38f},
            // one
            {0b0000000, 0b01111111, 0, 1.0f},
            // infinity
            {0b0000000, 0b11111111, 0, std::numeric_limits<float>::infinity()},
            // negative infinity
            {0b0000000, 0b11111111, 1, -std::numeric_limits<float>::infinity()},
        })

FP_TEST(tt_fp8e5,
        {
            {0b00, 0b00000, 0, 0.0f},
            {0b00, 0b00000, 1, -0.0f},
            // smallest positive subnormal number
            {0b01, 0b00000, 0, 0.0000152587890625f},
            // largest subnormal number
            {0b11, 0b00000, 0, 0.0000457763671875f},
            // smallest positive normal number
            {0b00, 0b00001, 0, 0.00006103515625f},
            // nearest value to 1/3
            {0b01, 0b01101, 0, 0.3125f},
            // largest number less than one
            {0b11, 0b01110, 0, 0.875f},
            // one
            {0b00, 0b01111, 0, 1.f},
            // smallest number larger than one
            // {0b01, 0b01111, 0, 1.25},
            // largest normal number
            {0b11, 0b11110, 0, 57344.f},
            // infinity
            {0b00, 0b11111, 0, std::numeric_limits<float>::infinity()},
            // negative infinity
            {0b00, 0b11111, 1, -std::numeric_limits<float>::infinity()},
        })

FP_TEST(tt_fp8e4nv,
        {
            {0b000, 0b0000, 0, 0.0f},
            {0b000, 0b0000, 1, -0.0f},
            // smallest positive subnormal number
            {0b001, 0b0000, 0, 0.001953125f},
            // largest subnormal number
            {0b111, 0b0000, 0, 0.013671875f},
            // smallest positive normal number
            {0b000, 0b0001, 0, 0.015625f},
            // nearest value to 1/3
            {0b011, 0b0101, 0, 0.34375f},
            // largest number less than one
            {0b111, 0b0110, 0, 0.9375f},
            // one
            {0b000, 0b0111, 0, 1.0f},
            // smallest number larger than one
            {0b001, 0b0111, 0, 1.125f},
            // largest normal number
            {0b111, 0b1110, 0, 240.0f},
            // infinity
            {0b000, 0b1111, 0, std::numeric_limits<float>::infinity()},
            // negative infinity
            {0b000, 0b1111, 1, -std::numeric_limits<float>::infinity()},
        })

FP_TEST(tt_fp8e4b15, {
                         {0b000, 0b0000, 0, 0.0f},
                         {0b000, 0b0000, 1, -0.0f},
                         // smallest positive subnormal number
                         {0b001, 0b0000, 0, 7.62939453125e-06f},
                         // largest subnormal number
                         {0b111, 0b0000, 0, 5.340576171875e-05f},
                         // smallest positive normal number
                         {0b000, 0b0001, 0, 6.103515625e-05f},
                         // nearest value to 1/3
                         {0b011, 0b1101, 0, 0.34375f},
                         // largest number less than one
                         {0b111, 0b1110, 0, 0.9375f},
                         // one
                         {0b000, 0b1111, 0, 1.0f},
                         // smallest number larger than one
                         {0b001, 0b1111, 0, 1.125f},
                         // largest normal number
                         {0b111, 0b1111, 0, 1.875f},
                     })

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
