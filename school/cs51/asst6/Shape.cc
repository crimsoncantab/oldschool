#include "Light.h"
#include "Shape.h"
#include "Scene.h"

Shape::Shape()
{
}

Shape::Shape(const Texture& texture)
    :texture(texture)
{
}

Texture Shape::GetTexture()
{
    return texture;
}
