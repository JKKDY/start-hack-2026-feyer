#include <gtest/gtest.h>

#include <string>

#include "dfm/version.hpp"

TEST(Version, ReturnsNonEmptyString) {
  const std::string v = dfm::version();
  EXPECT_FALSE(v.empty());
}
