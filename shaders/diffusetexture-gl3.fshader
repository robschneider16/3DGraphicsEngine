#version 130


uniform vec3 uLight, uColor;  // position and color of light
uniform sampler2D uTexUnit0;

in vec3 vNormal;
in vec3 vPosition;
in vec2 vTexCoord0; 
out vec4 fragColor;

void main() {

  vec3 tolight = normalize(uLight - vPosition);
  vec3 normal = normalize(vNormal);

  float diffuse = max(0.08, dot(normal, tolight));

  vec3 toV = -normalize(vec3(vPosition));
  vec3 h = normalize(toV + tolight);
  float specular = pow(max(0.0, dot(h, normal)), 64.0); 
  vec4 texColor0 = texture(uTexUnit0, vTexCoord0); // get texture color
  texColor0.x *= diffuse;
  texColor0.y *= diffuse;
  texColor0.z *= diffuse;
  fragColor =   0.4 * vec4(uColor * diffuse, 1.0) + 0.6 * texColor0 + vec4(vec3(0.6, 0.6, 0.6) * specular, 1); // mix color with texture0 mixed with difuse and specular
} 