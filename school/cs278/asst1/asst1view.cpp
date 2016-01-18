#include <iostream>
#include <cstdlib>
#include <cmath>

int main(int argc, char * argv[]) {
  if (argc == 3 || argc > 5) {
    std::cerr << "Usage: ./view <r> <g> <b> <max. pixel power> \n";
    std::cerr << "If max. pixel power is not given, then it is computed from the image. \n";
  }
  int x, y, r, g, b;
  r = g = b = 1;
  std::cin >> x >> y;
  float *m = new float[x * y], M(0);
  for (int i = 0; i < x * y; ++i) {
    std::cin >> m[i];
    M = std::max(M, m[i]);
  }
  std::cerr << "Maximum power at a pixel: " << M << std::endl;
  if (M == 0) M = 1;
  if (argc > 3) {
    r = (std::atoi(argv[1])) ? 1 : 0;
    g = (std::atoi(argv[2])) ? 1 : 0;
    b = (std::atoi(argv[3])) ? 1 : 0;
    if (argc == 5) M = std::atof(argv[4]);
  }
  else if (argc == 2) M = std::atof(argv[1]);
  std::cout << "P3 " << x << " " << y << " 255\n";
  for (int i = 0; i < x * y; ++i) {
    const int c = std::max(0, std::min(255, static_cast<int> (255.0 * m[i] / M)));
    std::cout << r * c << " " << g * c << " " << b * c << std::endl;
  }
  return 0;
}
