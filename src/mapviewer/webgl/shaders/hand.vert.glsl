#version 300 es

precision highp float;

layout(location = 0) in vec4 a_joint;

uniform mat4 u_viewProjMatrix;

out vec3 v_color;

void main() {
    gl_Position = u_viewProjMatrix * vec4(a_joint.xyz, 1.0);
    gl_PointSize = a_joint.w >= 2.0 ? 22.0 : 16.0;

    vec3 leftColor = vec3(0.1, 0.8, 1.0);
    vec3 rightColor = vec3(1.0, 0.5, 0.1);
    v_color = mix(leftColor, rightColor, mod(a_joint.w, 2.0));
    if (a_joint.w >= 2.0) {
        v_color = mix(v_color, vec3(1.0), 0.5);
    }
}
