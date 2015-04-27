

uniform vec3 uLight, uColor;   
uniform sampler2D uTexUnit0;

varying vec3 vNormal;
varying vec3 vPosition;
varying vec2 vTexCoord0;
//mediump vec4 gl_FragColor; 

void main() {
	
  vec3 tolight = normalize(uLight - vPosition);
  vec3 normal = normalize(vNormal);
  vec3 toV = -normalize(vec3(vPosition));
  vec3 h = normalize(toV + tolight);

  float diffuse = max(0.1, dot(normal, tolight));
  float specular = pow(max(0.0, dot(h, normal)), 64.0); 

  vec4 texColor0 = texture2D(uTexUnit0, vTexCoord0); // get texture color
  texColor0.x *= diffuse;
  texColor0.y *= diffuse;
  texColor0.z *= diffuse;
  gl_FragColor =   0.3 * vec4(uColor * diffuse, 1.0) + 0.7 * texColor0+ vec4(vec3(0.6, 0.6, 0.6) * specular, 1); // mix color with texture0 mixed with difuse

}
