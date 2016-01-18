#include <iostream>
using namespace std;

typedef struct {
    float x, y;
} point_t;

int main() {
    point_t p[3] = {{1,2},{3,4},{5,6}};
    point_t *pp;
    float *pf, *pf2;
    pp = &p[0];
    pf = &pp->x;
    pp++;
    *pf = pp->x;
    pp++;
    pp->y += p[2].y;
    cout << p[0].y << p[2].x;
    return 0;
}
