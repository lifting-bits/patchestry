#include "lib.hpp"

auto main() -> int
{
  auto const lib = library {};

  return lib.name == "patchestry" ? 0 : 1;
}
