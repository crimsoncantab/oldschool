//
// texmapping1.frag
//
// Make this do linear reconstruction regardless of which texturing mode 
// OpenGL is set to.
//

uniform float uTw;
uniform float uTh;
uniform sampler2D texUnit0;

varying vec2  pTexCoord;		// texture coordinates

void main()
{
  vec2 pTexelCoord;
  pTexelCoord.x = pTexCoord.x * uTw - .5;
  pTexelCoord.y = pTexCoord.y * uTh - .5;
  int intx = int(pTexelCoord.x);
  int inty = int(pTexelCoord.y);
  float fracx = pTexelCoord.x - intx;
  float fracy = pTexelCoord.y - inty;
  
  vec4 colorx1 = (1-fracx) * texture2D(texUnit0, vec2((intx + .5) / uTw, (inty+ .5) / uTh)) +
				 (fracx) *  texture2D(texUnit0, vec2((intx+ 1.5) / uTw, (inty+ .5) / uTh));
  vec4 colorx2 = (1-fracx) * texture2D(texUnit0, vec2((intx+ .5) / uTw, (inty+ 1.5) / uTh)) +
				 (fracx) *  texture2D(texUnit0, vec2((intx+ 1.5) / uTw, (inty+ 1.5) / uTh));
  gl_FragColor = (1-fracy)* colorx1 + (fracy) * colorx2;
}