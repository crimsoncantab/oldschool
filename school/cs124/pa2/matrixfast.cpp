/* 
 * File:   matrixmult.cpp
 * Author: loren
 *
 * Created on March 24, 2010, 3:56 PM
 */
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
//interleaving macro from
//http://graphics.stanford.edu/~seander/bithacks.html#InterleaveBMN
#define interleave(x,y,z) \
x = (x | (x << 8)) & 0x00FF00FF; \
x = (x | (x << 4)) & 0x0F0F0F0F; \
x = (x | (x << 2)) & 0x33333333; \
x = (x | (x << 1)) & 0x55555555; \
y = (y | (y << 8)) & 0x00FF00FF; \
y = (y | (y << 4)) & 0x0F0F0F0F; \
y = (y | (y << 2)) & 0x33333333; \
y = (y | (y << 1)) & 0x55555555; \
z = x | (y << 1)
//find the next power of two macro from
//http://acius2.blogspot.com/2007/11/calculating-next-power-of-2.html
#define pow2(val) \
val--; \
val = (val >> 1) | val; \
val = (val >> 2) | val; \
val = (val >> 4) | val; \
val = (val >> 8) | val; \
val = (val >> 16) | val; \
val++
#define BEST_CROSSOVER 8;
using namespace std;

enum Algorithm {
    DEFAULT, CONVENTIONAL, CACHE, STRASSEN, STRASSEN_MEM, HYBRID
};

enum DataGeneration {
    FROM_FILE, PLUS_MINUS, ZERO_TO_2, EMPTY
};

static int crossover, crossover_squared;

class SqMatrix {
private:

    enum DataLayout {
        ROW_MAJOR, COL_MAJOR, MORTON
    };

    bool should_del;
    int * data;
    DataLayout layout;
    int size;
public:

    const int dim;
    const int input_dim;

    SqMatrix(SqMatrix * m, int r_off_, int c_off_)
    : dim(m->dim / 2), input_dim(m->input_dim)
    {
        int i, j = r_off_ / crossover, k = c_off_ / crossover;
        interleave(j, k, i);
        i *= crossover_squared;
        data = &(m->data[i]);
        //        } else {
        //            data = m->data;
        //        }
        if (dim <= crossover) {
            layout = ROW_MAJOR;
        } else {
            layout = MORTON;
        }
        size = dim * dim;
    }

    SqMatrix(int input_dim_, int in_mem_dim_, int * data_)
    : data(data_), dim(in_mem_dim_), input_dim(input_dim_){
        if (dim <= crossover) {
            layout = ROW_MAJOR;
        } else {
            layout = MORTON;
        }
        size = dim * dim;

    }

    int getIndex(int r, int c) {
        int i, j, k;
        switch (layout) {
            case COL_MAJOR:
                i = c * dim + r;
                break;
            case MORTON:
                j = r / crossover;
                k = c / crossover;
                interleave(j, k, i);
                i *= crossover_squared;
                i += ((r % crossover) * crossover);
                i += (c % crossover);
                break;
            default:
                i = r * dim + c;
                break;
        }
        return i;
    }

    void setVal(int r, int c, int v) {
        data[getIndex(r, c)] = v;
    }

    void addVal(int r, int c, int v) {
        data[getIndex(r, c)] += v;
    }

    int getVal(int r, int c) {
        return data[getIndex(r, c)];
    }

    void setAsSumOf(SqMatrix & a, SqMatrix & b, bool add) {
        for (int i = 0; i < size; i++) {
            int val = a.data[i];
            if (add) val += b.data[i];
            else val -= b.data[i];
            data[i] = val;
        }
    }

    void printDiag() {
        for (int i = 0; i < input_dim; i++) {
            cout << getVal(i, i) << endl;
        }
    }

  /*  void printFull() {
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                cout << getVal(i, j) << "\t";
            }
            cout << endl;
        }
    }*/

    void load(ifstream & f) {
        for (int i = 0; i < input_dim; i++) {
            for (int j = 0; j < input_dim; j++) {
                int val;
                f >> val;
                setVal(i, j, val);
            }
        }
    }

    void populate(const DataGeneration mode) {
        for (int i = 0; i < input_dim; i++) {
            for (int j = 0; j < input_dim; j++) {
                int val;
                switch (mode) {
                    case FROM_FILE:
                        break;
                    case PLUS_MINUS:
                        val = (rand() % 3) - 1;
                        break;
                    case ZERO_TO_2:
                        val = rand() % 3;
                        break;
                    case EMPTY:
                        val = 0;
                        break;
                    default:
                        cout << "Bad enum DataGeneration" << endl;
                        exit(EXIT_FAILURE);
                }
                setVal(i, j, val);
            }
        }
    }

    void clear() {
        populate(EMPTY);
    }
};

void cachingConventionalMult(SqMatrix * a, SqMatrix * b, SqMatrix * c) {
    for (int i = 0; i < c->dim; i++) {
        for (int j = 0; j < c->dim; j++) {
            for (int k = 0; k < c->dim; k++) {
                if (i == 0) {//first time through, init c to 0s
                    c->setVal(j, k, 0);
                }
                c->addVal(j, k, a->getVal(j, i) * b->getVal(i, k));
            }
        }
    }
}

void hybrid(SqMatrix * A, SqMatrix * B, SqMatrix * C, SqMatrix * work) {
        if (C->dim < crossover) {cout<<"This is bad" << endl; exit(EXIT_FAILURE);}
    if (C->dim <= crossover) cachingConventionalMult(A, B, C);
    else {
        int dim2 = C->dim / 2;
        SqMatrix a(A, 0, 0), b(A, 0, dim2), c(A, dim2, 0), d(A, dim2, dim2),
                e(B, 0, 0), f(B, 0, dim2), g(B, dim2, 0), h(B, dim2, dim2),
                c11(C, 0, 0), c21(C, dim2, 0), c12(C, 0, dim2), c22(C, dim2, dim2),
                d11(work, 0, 0), d21(work, dim2, 0), d12(work, 0, dim2), d22(work, dim2, dim2);
        d11.setAsSumOf(f, h, false);
        hybrid(&a, &d11, &c12, &d22); // c12 = m1
        //        }
        d11.setAsSumOf(c, d, true);
        hybrid(&d11, &e, &c21, &d22); // c21 = m3
        d11.setAsSumOf(a, c, false);
        d12.setAsSumOf(e, f, true);
        hybrid(&d11, &d12, &c22, &d22); // c22 = m7
        d11.setAsSumOf(a, d, true);
        d12.setAsSumOf(e, h, true);
        hybrid(&d11, &d12, &c11, &d22); // c11 = m5
        c22.setAsSumOf(c11, c22, false); //c22 = m5 - m7
        c22.setAsSumOf(c22, c12, true); //c22 = m5 - m7 + m1
        c22.setAsSumOf(c22, c21, false); //c22 = m5 - m7 + m1 - m3 (done with this quadrant)
        d11.setAsSumOf(a, b, true);
        hybrid(&d11, &h, &d12, &d22); // d22 = m2
        c12.setAsSumOf(c12, d12, true); // c12 = m1 + m2 (done with this quadrant)
        c11.setAsSumOf(c11, d12, false); // c11 = m5 - m2
        d11.setAsSumOf(g, e, false);
        hybrid(&d, &d11, &d12, &d22); // d12 = m4
        c21.setAsSumOf(c21, d12, true); // c21 = m3 + m4 (done with this quadrant)
        c11.setAsSumOf(c11, d12, true); // c11 = m5 - m2 + m4
        d11.setAsSumOf(b, d, false);
        d12.setAsSumOf(g, h, true);
        hybrid(&d11, &d12, &d21, &d22); // d21 = m6
        c11.setAsSumOf(c11, d21, true); // c11 = m5 - m2 + m4 + m6
    }
}

int calc_opt_padding(int dim) {
    int next2 = dim;
    pow2(next2);
    if (dim == next2) { //we're done, make sure crossover is padded
        pow2(crossover);
	return dim;
    } else if (crossover > dim) { //we're done
        crossover = dim;
        return dim;
    } else {
        int tempcrossover = crossover;
        int crossover2 = crossover << 1;
        int diff = -1;
        int best_dim;
        int best_crossover;
        do {
            int temp = tempcrossover;
            while (temp < dim) {
                temp = temp << 1;
            }
            if ((temp - dim) < diff || diff == -1) {
                diff = temp - dim;
                best_crossover = tempcrossover;
                best_dim = temp;
            }
            tempcrossover++;
        } while (tempcrossover < crossover2);
        crossover = best_crossover;
        return best_dim;
    }
}

// run hybrid, load from file

void run_normal_mode(int argc, char** argv) {
    if (atoi(argv[1]) == 0) {
        crossover = BEST_CROSSOVER;
    } else crossover = atoi(argv[3]);

    int dim = atoi(argv[2]);
    int newdim = calc_opt_padding(dim);
//    cout<<"new dim: " << newdim << " xover: " << crossover << endl;
    crossover_squared = crossover * crossover;
    int dim2 = newdim * newdim;
    int * data = new int[dim2 * 4];
    SqMatrix a(dim, newdim, data);
    SqMatrix b(dim, newdim, &(data[dim2]));
    SqMatrix c(dim, newdim, &(data[dim2 * 2]));
    SqMatrix d(dim, newdim, &(data[dim2 * 3]));

    if (atoi(argv[1]) == 0) {
        ifstream input(argv[3]);
        a.load(input);
        b.load(input);
        input.close();
    } else {
        srand(time(NULL));
        a.populate(PLUS_MINUS);
        b.populate(PLUS_MINUS);
    }
    clock_t pre, post;
    pre = clock();
    hybrid(&a, &b, &c, &d);
    post = clock();
    if (atoi(argv[1])==0) {
        c.printDiag();
    } else cout << double(post - pre) / (double) CLOCKS_PER_SEC << endl;
    delete[] data;
}


int main(int argc, char** argv) {
    run_normal_mode(argc, argv);
    return EXIT_SUCCESS;
}

