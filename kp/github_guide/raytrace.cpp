#include "ray.h"
#include "random.h"
#include "camera.h"
#include "hittable_list.h"
#include "sphere.h"
#include <iostream>

bool intersectPlane(const vec3 &n, const vec3 &p0, const vec3 &l0, const vec3 &l, float& t) {
    // assuming vectors are all normalized
    float denom = dot(n, l);
    if (denom > 1e-6) {
        vec3 p0l0 = p0 - l0;
        t = dot(p0l0, n) / denom;
        return (t >= 0);
    }

    return false;
}
bool rayTriangleIntersect(
    const vec3 &orig, const vec3 &dir,
    const vec3 &v0, const vec3 &v1, const vec3 &v2,
    float &t)
{
    float kEpsilon = 1e-7;
    // compute plane's normal
    vec3 v0v1 = v1 - v0;
    vec3 v0v2 = v2 - v0;
    // no need to normalize
    vec3 N = cross(v0v1, v0v2); // N
    float area2 = N.length();

    // Step 1: finding P

    // check if ray and plane are parallel ?
    float NdotRayDirection = dot(N, dir);
    if (fabs(NdotRayDirection) < kEpsilon) // almost 0
        return false; // they are parallel so they don't intersect !

    // compute d parameter using equation 2
    float d = dot(N, v0);

    // compute t (equation 3)
    t = (dot(N, orig) + d) / NdotRayDirection;
    // check if the triangle is in behind the ray
    if (t < 0) return false; // the triangle is behind

    // compute the intersection point using equation 1
    vec3 P = orig + t * dir;

    // Step 2: inside-outside test
    vec3 C; // vector perpendicular to triangle's plane

    // edge 0
    vec3 edge0 = v1 - v0;
    vec3 vp0 = P - v0;
    C = cross(edge0, vp0);
    if (dot(N, C) < 0) return false; // P is on the right side

    // edge 1
    vec3 edge1 = v2 - v1;
    vec3 vp1 = P - v1;
    C = cross(edge1, vp1);
    if (dot(N, C) < 0)  return false; // P is on the right side

    // edge 2
    vec3 edge2 = v0 - v2;
    vec3 vp2 = P - v2;
    C = cross(edge2, vp2);
    if (dot(N, C) < 0) return false; // P is on the right side;

    return true; // this ray hits the triangle
}

float color_at_dist(float t){
    // std::cerr << "/* error message */" << std::min(1-t/50, 1.f)<<'\n';
    return std::max(0.f, std::min(1-t/2, 1.f));
}
int main() {
    int nx = 200;
    int ny = 100;
    int ns = 100;
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";

    // hittable *list[2];
    // list[0] = new sphere(vec3(0,0,-1), 0.5);
    // list[1] = new sphere(vec3(0,-100.5,-1), 100);
    // hittable *world = new hittable_list(list,2);
    camera cam;
    vec3 p0(0, 0, -1); // how far plane from the origin
    p0.make_unit_vector();
    vec3 n(0, -0.75, -0.5); // plane normal
    n.make_unit_vector(); // normalize

    vec3 v0(0, 0.5, -0.5);
    vec3 v1(0.5, 0.5, -0.5);
    vec3 v2(0.5, -0.5, -0.1);

    float max_t = 0;

    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            vec3 col(1, 1, 1);
            vec3 white(1, 1, 1);
            vec3 blue(0.75, 1, 1);
            vec3 green(0, 1, 0);
            vec3 red(1, 0, 0);
            float u = float(i) / float(nx);
            float v = float(j) / float(ny);
            ray r = cam.get_ray(u, v);
            vec3 l0 = r.origin();
            vec3 l = r.direction();
            float t = 0;


            if(rayTriangleIntersect(l0, l,
                v0, v1, v2, t)){
                float sum_dist = t;
                col = green + 2.2*color_at_dist(t)*red;
                vec3 new_l0 = r.point_at_parameter(t);
                if (intersectPlane(n, p0, new_l0, l, t)){
                    sum_dist += t;
                    col /=2 ;
                    col += (vec3(1, 1, 1) * color_at_dist(sum_dist))/2 ;
                    // col.make_unit_vector();
                }
            } else {

                if (intersectPlane(n, p0, l0, l, t))
                    col = white * color_at_dist(t);
                else
                    col = blue;
            }
            // std::cerr << "t:" << t << '\n';
            int ir = int(255.99*col[0]);
            int ig = int(255.99*col[1]);
            int ib = int(255.99*col[2]);

            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
    std::cerr << "t:" << max_t << '\n';

}
