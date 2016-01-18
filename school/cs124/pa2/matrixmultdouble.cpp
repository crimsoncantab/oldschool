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
using namespace std;

enum Algorithm {
    DEFAULT, CONVENTIONAL, CACHE, STRASSEN, STRASSEN_MEM, HYBRID
};

enum DataGeneration {
    FROM_FILE, PLUS_MINUS, ZERO_TO_2, REALS, EMPTY
};

class SqMatrix {
private:

    bool should_del;
    double * data;
    const int c_off;
    const int r_off;
    int data_split;
public:

    enum DataLayout {
        ROW_MAJOR, COL_MAJOR, MORTON
    };
    int dim;
    const int real_dim;
    const DataLayout layout;

    SqMatrix(SqMatrix * m, int r_off_, int c_off_)
    : data_split(m->data_split), dim(m->dim / 2), real_dim(m->real_dim), layout(m->layout), c_off(m->c_off + c_off_),
    r_off(m->r_off + r_off_) {
        if (layout == MORTON) {
            int i, j = r_off_, k = c_off_;
            interleave(j, k, i);
            data = &(m->data[i]);
        } else {
            data = m->data;
        }
        should_del = false;
    }

    SqMatrix(int dim_, double * data_, DataLayout data_layout_) : dim(dim_), real_dim(dim_), c_off(0), r_off(0), layout(data_layout_) {
        pow2(dim);
        data = data_;
        data_split = dim;
        should_del = false;
    }

    SqMatrix(int dim_) : dim(dim_), real_dim(dim_), c_off(0), r_off(0), layout(ROW_MAJOR) {
        pow2(dim);
        data = new double[dim * dim];
        data_split = dim;
        should_del = true;
    }

    SqMatrix(int dim_, DataLayout data_layout_) : dim(dim_), real_dim(dim_), c_off(0), r_off(0), layout(data_layout_) {
        pow2(dim);
        data = new double[dim * dim];
        data_split = dim;
        should_del = true;
    }

    ~SqMatrix() {
        if (should_del) {
            delete[] data;
        }
    }

    void setVal(int r, int c, double v) {
        int i;
        switch (layout) {
            case COL_MAJOR:
                c += c_off;
                r += r_off;
                i = c * data_split + r;
                break;
            case MORTON:
                interleave(r, c, i);
                break;
            default:
                c += c_off;
                r += r_off;
                i = r * data_split + c;
                break;
        }
        data[i] = v;
    }

    void addVal(int r, int c, double v) {
        int i;
        switch (layout) {
            case COL_MAJOR:
                c += c_off;
                r += r_off;
                i = c * data_split + r;
                break;
            case MORTON:
                interleave(r, c, i);
                break;
            default:
                c += c_off;
                r += r_off;
                i = r * data_split + c;
                break;
        }
        data[i] += v;
    }

    double getVal(int r, int c) {
        int i;
        switch (layout) {
            case COL_MAJOR:
                c += c_off;
                r += r_off;
                i = c * data_split + r;
                break;
            case MORTON:
                interleave(r, c, i);
                break;
            default:
                c += c_off;
                r += r_off;
                i = r * data_split + c;
                break;
        }
        return data[i];
    }

    bool isAllPadding() {
        return (real_dim <= c_off || real_dim <= r_off);
    }
    //optimized for morton layout

    void setAsSumOf(SqMatrix & a, SqMatrix & b, bool add) {
        if (layout == MORTON && a.layout == MORTON && b.layout == MORTON) {
            int size = dim * dim;
            for (int i = 0; i < size; i++) {
                double val = a.data[i]; // + add ? b.data[i] : -(b.data[i]);
                if (add) val += b.data[i];
                else val -= b.data[i];
                data[i] = val;
            }
        } else {
            for (int i = 0; i < dim; i++) {
                for (int j = 0; j < dim; j++) {
                    double val = a.getVal(i, j);
                    if (add) val += b.getVal(i, j);
                    else val -= b.getVal(i, j);
                    setVal(i, j, val);
                }
            }
        }
    }

    void printDiag() {
        for (int i = 0; i < real_dim; i++) {
            cout << getVal(i, i) << endl;
        }
    }

    void printFull() {
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                cout << getVal(i, j) << "\t";
            }
            cout << endl;
        }
    }

    void load(ifstream & f) {
        for (int i = 0; i < real_dim; i++) {
            for (int j = 0; j < real_dim; j++) {
                double val;
                f >> val;
                setVal(i, j, val);
            }
        }
    }

    void populate(const DataGeneration mode) {
        for (int i = 0; i < real_dim; i++) {
            for (int j = 0; j < real_dim; j++) {
                double val;
                switch (mode) {
                    case FROM_FILE:
                        break;
                    case PLUS_MINUS:
                        val = (rand() % 3) - 1;
                        break;
                    case ZERO_TO_2:
                        val = rand() % 3;
                        break;
                    case REALS:
                        val = rand() / (double) RAND_MAX;
                        break;
                    case EMPTY:
                        val = 0.;
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
static SqMatrix::DataLayout my_layout = SqMatrix::MORTON;
static int crossover = 1;

void naiveConventionalMult(SqMatrix * a, SqMatrix * b, SqMatrix * c) {
    //assumes all values in c are init to 0
    for (int i = 0; i < c->dim; i++) {
        for (int j = 0; j < c->dim; j++) {
            c->setVal(i, j, 0);
            for (int k = 0; k < c->dim; k++) {
                c->addVal(i, j, a->getVal(i, k) * b->getVal(k, j));
            }
        }
    }
}

void cachingConventionalMult(SqMatrix * a, SqMatrix * b, SqMatrix * c) {
    //assumes all values in c are init to 0
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

void naiveStrassensMult(SqMatrix * A, SqMatrix * B, SqMatrix * C) {
    if (C->dim == 1) C->setVal(0, 0, A->getVal(0, 0) * B->getVal(0, 0));
    else {
        int dim2 = C->dim / 2;
        SqMatrix a(A, 0, 0), b(A, 0, dim2), c(A, dim2, 0), d(A, dim2, dim2),
                e(B, 0, 0), f(B, 0, dim2), g(B, dim2, 0), h(B, dim2, dim2),
                c11(C, 0, 0), c21(C, dim2, 0), c12(C, 0, dim2), c22(C, dim2, dim2),
                tmp1(dim2, my_layout), tmp2(dim2, my_layout),
                m1(dim2, my_layout), m2(dim2, my_layout), m3(dim2, my_layout),
                m4(dim2, my_layout), m5(dim2, my_layout), m6(dim2, my_layout),
                m7(dim2, my_layout);
        tmp1.setAsSumOf(f, h, false);
        naiveStrassensMult(&a, &tmp1, &m1);
        tmp1.setAsSumOf(a, b, true);
        naiveStrassensMult(&tmp1, &h, &m2);
        tmp1.setAsSumOf(c, d, true);
        naiveStrassensMult(&tmp1, &e, &m3);
        tmp1.setAsSumOf(g, e, false);
        naiveStrassensMult(&d, &tmp1, &m4);
        tmp1.setAsSumOf(a, d, true);
        tmp2.setAsSumOf(e, h, true);
        naiveStrassensMult(&tmp1, &tmp2, &m5);
        tmp1.setAsSumOf(b, d, false);
        tmp2.setAsSumOf(g, h, true);
        naiveStrassensMult(&tmp1, &tmp2, &m6);
        tmp1.setAsSumOf(a, c, false);
        tmp2.setAsSumOf(e, f, true);
        naiveStrassensMult(&tmp1, &tmp2, &m7);
        c11.setAsSumOf(m5, m4, true);
        c11.setAsSumOf(c11, m2, false);
        c11.setAsSumOf(c11, m6, true);
        c12.setAsSumOf(m1, m2, true);
        c21.setAsSumOf(m3, m4, true);
        c22.setAsSumOf(m5, m1, true);
        c22.setAsSumOf(c22, m3, false);
        c22.setAsSumOf(c22, m7, false);

    }
}

void fasterStrassensMult(SqMatrix * A, SqMatrix * B, SqMatrix * C, SqMatrix * work) {
    if (C->dim == 1) C->setVal(0, 0, A->getVal(0, 0) * B->getVal(0, 0));
    else {
        int dim2 = C->dim / 2;
        SqMatrix a(A, 0, 0), b(A, 0, dim2), c(A, dim2, 0), d(A, dim2, dim2),
                e(B, 0, 0), f(B, 0, dim2), g(B, dim2, 0), h(B, dim2, dim2),
                c11(C, 0, 0), c21(C, dim2, 0), c12(C, 0, dim2), c22(C, dim2, dim2),
                d11(work, 0, 0), d21(work, dim2, 0), d12(work, 0, dim2), d22(work, dim2, dim2);
        d11.setAsSumOf(f, h, false);
        fasterStrassensMult(&a, &d11, &c12, &d22); // c12 = m1
        d11.setAsSumOf(c, d, true);
        fasterStrassensMult(&d11, &e, &c21, &d22); // c21 = m3
        d11.setAsSumOf(a, c, false);
        d12.setAsSumOf(e, f, true);
        fasterStrassensMult(&d11, &d12, &c22, &d22); // c22 = m7
        d11.setAsSumOf(a, d, true);
        d12.setAsSumOf(e, h, true);
        fasterStrassensMult(&d11, &d12, &c11, &d22); // c11 = m5
        c22.setAsSumOf(c11, c22, false); //c22 = m5 - m7
        c22.setAsSumOf(c22, c12, true); //c22 = m5 - m7 + m1
        c22.setAsSumOf(c22, c21, false); //c22 = m5 - m7 + m1 - m3 (done with this quadrant)
        d11.setAsSumOf(a, b, true);
        fasterStrassensMult(&d11, &h, &d12, &d22); // d22 = m2
        c12.setAsSumOf(c12, d12, true); // c12 = m1 + m2 (done with this quadrant)
        c11.setAsSumOf(c11, d12, false); // c11 = m5 - m2
        d11.setAsSumOf(g, e, false);
        fasterStrassensMult(&d, &d11, &d12, &d22); // d12 = m4
        c21.setAsSumOf(c21, d12, true); // c21 = m3 + m4 (done with this quadrant)
        c11.setAsSumOf(c11, d12, true); // c11 = m5 - m2 + m4
        d11.setAsSumOf(b, d, false);
        d12.setAsSumOf(g, h, true);
        fasterStrassensMult(&d11, &d12, &d21, &d22); // d21 = m6
        c11.setAsSumOf(c11, d21, true); // c11 = m5 - m2 + m4 + m6
    }
}

void hybrid(SqMatrix * A, SqMatrix * B, SqMatrix * C, SqMatrix * work) {
    if (C->dim <= crossover) cachingConventionalMult(A, B, C);
    else {
        int dim2 = C->dim / 2;
        SqMatrix a(A, 0, 0), b(A, 0, dim2), c(A, dim2, 0), d(A, dim2, dim2),
                e(B, 0, 0), f(B, 0, dim2), g(B, dim2, 0), h(B, dim2, dim2),
                c11(C, 0, 0), c21(C, dim2, 0), c12(C, 0, dim2), c22(C, dim2, dim2),
                d11(work, 0, 0), d21(work, dim2, 0), d12(work, 0, dim2), d22(work, dim2, dim2);
        d11.setAsSumOf(f, h, false);
        hybrid(&a, &d11, &c12, &d22); // c12 = m1
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

// run hybrid, load from file

int run_normal_mode(int argc, char** argv) {
    my_layout = SqMatrix::MORTON;

    int dim = atoi(argv[2]);
    int next2 = dim;
    pow2(next2);
    int dim2 = next2 * next2;
    double * data = new double[dim2 * 4];
    SqMatrix a(dim, data, my_layout);
    SqMatrix b(dim, &(data[dim2]), my_layout);
    SqMatrix c(dim, &(data[dim2 * 2]), my_layout);
    SqMatrix d(dim, &(data[dim2 * 3]), my_layout);
    ifstream input(argv[3]);
    a.load(input);
    b.load(input);
    input.close();
    hybrid(&a, &b, &c, &d);
    c.printDiag();
    delete[] data;
}

int run_experimental_mode(int argc, char** argv) {

    //strassen mode dim data_gen layout crossover suppress_output [file]
    Algorithm mode = (Algorithm) atoi(argv[1]);
    int dim = atoi(argv[2]);
    int next2 = dim;
    pow2(next2);
    int dim2 = next2 * next2;
    double * data;
    if (mode == HYBRID || mode == STRASSEN_MEM) {
        data = new double[dim2 * 4];
    } else {
        data = new double[dim2 * 3];
    }
    DataGeneration data_gen_mode = (DataGeneration) atoi(argv[3]);
    my_layout = (SqMatrix::DataLayout)atoi(argv[4]);
    crossover = atoi(argv[5]);
    bool verbose_output = (atoi(argv[6]) != 0);
    SqMatrix a(dim, data, my_layout);
    SqMatrix b(dim, &(data[dim2]), my_layout);
    SqMatrix c(dim, &(data[dim2 * 2]), my_layout);
    if (data_gen_mode == FROM_FILE) {
        ifstream input(argv[7]);
        a.load(input);
        b.load(input);
        input.close();
    } else {
        srand(time(NULL));
        a.populate(data_gen_mode);
        b.populate(data_gen_mode);
    }
    if (verbose_output) {
        cout << "a" << endl;
        a.printFull();
        cout << "b" << endl;
        b.printFull();
    }
    clock_t pre, post;
    string alg_name;
    pre = clock();
    switch (mode) {
        case DEFAULT:
            exit(EXIT_FAILURE);
            break;
        case CONVENTIONAL:
            alg_name = "Conventional";
            naiveConventionalMult(&a, &b, &c);
            break;
        case CACHE:
            alg_name = "Cache Aware Conventional";
            cachingConventionalMult(&a, &b, &c);
            break;
        case STRASSEN:
            alg_name = "Strassen's";
            naiveStrassensMult(&a, &b, &c);
            break;
        case STRASSEN_MEM:
            alg_name = "Mem Efficient Strassen's";
        {
            SqMatrix d(dim, &(data[dim2 * 3]), my_layout);
            fasterStrassensMult(&a, &b, &c, &d);
        }
            break;
        case HYBRID:
            alg_name = "Hybrid";
        {
            SqMatrix d(dim, &(data[dim2 * 3]), my_layout);
            hybrid(&a, &b, &c, &d);
        }
            break;
        default:
            cout << "Bad Enum Algorithm" << endl;
            exit(EXIT_FAILURE);
    }

    post = clock();
    if (verbose_output) {
        string layout_name;
        switch (my_layout) {
            case SqMatrix::MORTON:
                layout_name = "Morton";
                break;
            case SqMatrix::COL_MAJOR:
                layout_name = "Column Major";
                break;
            case SqMatrix::ROW_MAJOR:
                layout_name = "Default";
                break;
            default:
                layout_name = "Unknown";
                break;
        }
        cout << "Data Layout: " << layout_name << endl;
        cout << alg_name << " Runtime: ";
    }
    delete data;
    cout << double(post - pre) / (double) CLOCKS_PER_SEC << endl;
    if (verbose_output) c.printFull();
}

int main(int argc, char** argv) {
    cout.precision(3);
    Algorithm mode = (Algorithm) atoi(argv[1]);
    if (mode == DEFAULT) {
        run_normal_mode(argc, argv);
    } else {
        run_experimental_mode(argc, argv);
    }
    return EXIT_SUCCESS;
}

