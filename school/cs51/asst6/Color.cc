#include "Color.h"

Color::Color()
    :r(0), g(0), b(0)
{
}

Color::Color(double red, double green, double blue)
    :r(red), g(green), b(blue)
{
    Clamp();
}

void Color::Clamp()
{
    r = r > 1.0 ? 1.0 : (r < 0.0 ? 0.0 : r);
    g = g > 1.0 ? 1.0 : (g < 0.0 ? 0.0 : g);
    b = b > 1.0 ? 1.0 : (b < 0.0 ? 0.0 : b);
}

Color operator * (double s, const Color& c) { return c * s; }

std::ofstream& operator <<(std::ofstream& fs, const Color& c)
{
    Color copy = c;
    copy.Clamp();
    
    char r = (char)(copy.GetR() * 255);
    char g = (char)(copy.GetG() * 255);
    char b = (char)(copy.GetB() * 255);
    
    fs << b << g << r;
    return fs;
}
