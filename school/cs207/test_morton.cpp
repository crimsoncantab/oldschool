#include "MortonCoder.hpp"
#include "BoundingBox.hpp"

#include "CS207/Util.hpp"

using namespace std;

int main() {
  MortonCoder<1> m(BoundingBox(Point(0,0,0), Point(1,1,1)));
  BoundingBox region(Point(.25, .25, .25), Point(.25, .75, .75));
  for (auto it = m.cell_begin(region); it != m.cell_end(region); ++it) {
    cout << *it << endl;
  }

  return 0;
}
