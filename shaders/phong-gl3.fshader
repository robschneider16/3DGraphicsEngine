#version 130
uniform vec3 uLight, uColor; 
 
in vec3 vNormal; // normal to surface
in vec3 vPosition; // position of point on surface
 
out vec4 fragColor;
 
void main() {
	vec3 toLight, normal, Col, h, b, toV;
	if(gl_FrontFacing == false){
	  toLight = uLight - vec3(vPosition);
	  toLight = normalize(toLight);
	  normal = normalize(vec3(-vNormal[0], -vNormal[1], -vNormal[2]));
	  Col = vec3(uColor[1], uColor[2], uColor[0]);
	}else{
	  toLight = uLight - vec3(vPosition);
	  toLight = normalize(toLight);
	  normal = normalize(vNormal);
	  Col = uColor;
	  }
	toV = -normalize(vec3(vPosition));
	h = normalize(toV + toLight);
	  
	float specular = pow(max(0.0, dot(h, normal)), 64.0);  
	float diffuse = max(0.0, dot(normal, toLight));
	vec3 intensity =  vec3(0.1, 0.1, 0.1) + Col * diffuse + (vec3(0.6, 0.6, 0.6) * specular);

	fragColor = vec4(intensity.x, intensity.y, intensity.z, 1.0);
}
