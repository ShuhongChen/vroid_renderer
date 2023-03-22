#version 330

#if defined VERTEX_SHADER

in vec3 in_position;
in vec3 in_normal;
in vec2 in_texcoord_0;

uniform mat4 m_proj;
uniform mat4 m_model;
uniform mat4 m_cam;
uniform vec3 loc_origin;

out vec3 normal;
out vec2 uv;
out vec3 pos;
out vec3 posxyz;

void main() {
    mat4 mv = m_cam * m_model; 
    vec4 p = mv * vec4(in_position, 1.0);
	gl_Position = m_proj * p;
    mat3 m_normal = transpose(inverse(mat3(mv)));
    normal = m_normal * in_normal;
    uv = in_texcoord_0;
    pos = p.xyz;
    posxyz = (m_model * vec4(in_position, 1.0)).xyz - loc_origin;
}

#elif defined FRAGMENT_SHADER

out vec4 fragColor;
uniform sampler2D texture0;

in vec3 normal;
in vec3 pos;
in vec3 posxyz;
in vec2 uv;

uniform vec4 base_color_factor;


void main()
{
    float l = dot(normalize(-pos), normalize(normal));
    vec4 color = texture(texture0, uv) * base_color_factor;
    if (color.a < 0.5) {
        discard;
    }
    // color = vec4(color.xyz, 0.5);
    // fragColor = color;
    fragColor = vec4((posxyz.xyz+0.25)*2, color.a);
    // fragColor = vec4(posxyz.xyz, color.a);

    // fragColor = vec4(color.xyz, 1.0);
    // fragColor = color * 0.25 + color * 0.75 * abs(l);
    // fragColor = color;
    // fragColor = vec4(abs(l),abs(l),abs(l),1.0);
    // fragColor = vec4(1.0);
    // fragColor = vec4(-pos.z,-pos.z,-pos.z, 1.0);
    // fragColor = vec4(uv.x, uv.y, 1.0, 1.0);
    // if (l<0.5) {
    //     fragColor = vec4(0,0,0,1);
    // } else {
    //     fragColor = color;
    // }
    // fragColor = vec4(dFdx(normal), 1);
    // if (normalize(dFdx(pos)) > 0.1) {
    // if (dFdx(pos).x > 0.001) {
    //     fragColor = vec4(0,0,0,1);
    // }
}

#endif
