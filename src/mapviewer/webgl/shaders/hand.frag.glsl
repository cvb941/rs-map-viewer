#version 300 es

precision highp float;

in vec3 v_color;

layout(location = 0) out vec4 fragColor;

void main() {
    float distanceFromCenter = length(gl_PointCoord - 0.5);
    if (distanceFromCenter > 0.5) {
        discard;
    }
    fragColor = vec4(v_color, 1.0 - smoothstep(0.35, 0.5, distanceFromCenter));
}
