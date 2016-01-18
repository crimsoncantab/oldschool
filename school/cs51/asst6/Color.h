#ifndef _COLOR_
#define _COLOR_

#include <fstream>

class Color
{
 private:
    double r, g, b;

 public:
    Color();
    Color(double r, double g, double b);
    
    double GetR() const { return r; }
    double GetG() const { return g; }
    double GetB() const { return b; }
    
    Color operator += (const Color& c) { r += c.r; g += c.g; b += c.b; return *this; }
    Color operator + (const Color& c) const { return Color(*this) += c; }
    
    Color operator -= (const Color& c) { r -= c.r; g -= c.g; b -= c.b; return *this; }
    Color operator - (const Color& c) const { return Color(*this) -= c; }
    
    Color operator *= (const double s) { r *= s; g *= s; b *= s; return *this; }
    Color operator * (const double s) const { return Color(*this) *= s; }
    
    Color operator /= (const double s) { r /= s; g /= s; b /= s; return *this; }
    Color operator / (const double s) const { return Color(*this) /= s; }
    
    void Clamp();
};

Color operator * (double s, const Color& c);
std::ofstream& operator <<(std::ofstream& fs, const Color& c);

#endif //_COLOR_
