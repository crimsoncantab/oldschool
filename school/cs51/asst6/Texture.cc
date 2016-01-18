#include "Texture.h"

Texture::Texture()
{
}

Texture::Texture(const Color& baseColor, double ambientCoeff)
{
    base = baseColor;
    ambient = ambientCoeff;
}

