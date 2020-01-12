in vec2 TexCoords;
in vec3 WorldPos;
in vec3 Normal;
  
void main()
{
    vec3 normalized_normal = normalize(Normal);
    gl_FragColor = vec4(normalized_normal, 1.0);
}