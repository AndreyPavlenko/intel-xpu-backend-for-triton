#include <bitset>
#include <limits>

#include <gtest/gtest.h>

#include "third_party/intel/backend/reproducer/KernelLauncher.cpp"

TEST(KernelLauncherTest, test_tt_fp16) {
  struct TestCase {
    uint32_t mantissa : 10;
    uint32_t exponent : 5;
    uint32_t sign : 1;
    float expected;
  };
  TestCase testCases[]{
      {0b0000000000, 0b00000, 0, 0.0f},
      {0b0000000000, 0b00000, 1, -0.0f},
      // smallest positive subnormal number
      {0b0000000001, 0b00000, 0, 0.000000059604645},
      // largest subnormal number
      {0b1111111111, 0b00000, 0, 0.000060975552},
      // smallest positive normal number
      {0b0000000000, 0b00001, 0, 0.00006103515625},
      // nearest value to 1/3
      {0b0101010101, 0b01101, 0, 0.33325195},
      // largest number less than one
      {0b1111111111, 0b01110, 0, 0.99951172},
      // one
      {0b0000000000, 0b01111, 0, 1},
      // smallest number larger than one
      {0b0000000001, 0b01111, 0, 1.00097656},
      // largest normal number
      {0b1111111111, 0b11110, 0, 65504},
      // infinity
      {0b0000000000, 0b11111, 0, std::numeric_limits<float>::infinity()},
      // negative infinity
      {0b0000000000, 0b11111, 1, -std::numeric_limits<float>::infinity()},
      {0b0000000000, 0b10000, 1, -2},
      {0b0000000010, 0b00011, 0, 0.00024462890625},
  };
  for (const auto &[mantissa, exponent, sign, expected] : testCases) {
    tt_fp16 half = {mantissa, exponent, sign};
    const float result = static_cast<float>(half);
    ASSERT_TRUE(isClose(result, expected))
        << "Conversion from tt_fp16 to float failed! Mantissa: "
        << std::bitset<10>(mantissa).to_string()
        << ". Exponent: " << std::bitset<5>(exponent).to_string()
        << ". Sign: " << sign << ", Expected: " << expected
        << ". Actual: " << result << '.';
  }
}

TEST(KernelLauncherTest, test_tt_bf16) {
  struct TestCase {
    uint32_t mantissa : 7;
    uint32_t exponent : 8;
    uint32_t sign : 1;
    float expected;
  };
  TestCase testCases[] = {
      // zero
      {0b0000000, 0b00000000, 0, 0.0f},
      // negative zero
      {0b0000000, 0b00000000, 1, -0.0f},
      // smallest positive normal number
      {0b0000000, 0b00000001, 0, 1.17549e-38},
      // one
      {0b0000000, 0b01111111, 0, 1.0f},
      // infinity
      {0b0000000, 0b11111111, 0, std::numeric_limits<float>::infinity()},
      // negative infinity
      {0b0000000, 0b11111111, 1, -std::numeric_limits<float>::infinity()},
  };
  for (const auto &[mantissa, exponent, sign, expected] : testCases) {
    tt_bf16 bfloat = {mantissa, exponent, sign};
    float result = static_cast<float>(bfloat);
    ASSERT_TRUE(isClose(result, expected))
        << "Conversion from tt_bf16 to float failed! Mantissa: "
        << std::bitset<7>(mantissa).to_string()
        << ". Exponent: " << std::bitset<8>(exponent).to_string()
        << ". Sign: " << sign << ", Expected: " << expected
        << ". Actual: " << result << '.';
  }
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
