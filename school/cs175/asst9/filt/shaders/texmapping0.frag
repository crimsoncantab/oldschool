
uniform float uTw;
uniform float uTh;
uniform sampler2D texUnit0;

varying vec2  pTexCoord;		// texture coordinates

void main()
{
  vec4 texColor0 = texture2D(texUnit0, pTexCoord);
  gl_FragColor = texColor0;
}
