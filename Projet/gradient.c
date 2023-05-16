#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "time.h"

double zone[4] = {0, 4, 0, 4};
int DECOUPAGE = 100;
int MAX_ITER = 10000;

double distance(double* points, int nb, double x, double y) {

    double min = INFINITY;

    for (int i = 0; i < nb; i++) {
        double dx = points[2 * i] - x;
        double dy = points[2 * i + 1] - y;
        double d = sqrt(dx * dx + dy * dy);
        if (d < min) {
            min = d;
        }
    }

    return min;

}

double simpson(double* points, int nb, double x0, double x1, double y0, double y1, int n) {

    double h = (x1 - x0) / n;
    double k = (y1 - y0) / n;

    double integral = 0;

    for (int i = 0; i <= n; i++) {

        double x = x0 + i * h;
        double wx = (i == 0 || i == n) ? 1 : (i % 2 == 1) ? 4 : 2;

        for (int j = 0; j <= n; j++) {

            double y = y0 + j * k;
            double wy = (j == 0 || j == n) ? 1 : (j % 2 == 1) ? 4 : 2;

            integral += wx * wy * distance(points, nb, x, y);
        }

    }

    integral *= h * k / 9.0;
    return integral;
}

double objectif(double* points, int nb, int decoupage) {
    return simpson(points, nb, zone[0], zone[1], zone[2], zone[3], decoupage)/(zone[1]-zone[0])/(zone[3]-zone[2]);
}

double* gradient(double* points, int nb) {
    
    double h = 0.00001;
    double* grad = malloc(2 * nb * sizeof(double));
    double y = objectif(points, nb, DECOUPAGE);
    for (int i = 0; i < nb; i++) {
        points[2 * i] += h;
        grad[2 * i] = (objectif(points, nb, DECOUPAGE) - y) / h;
        points[2 * i] -= h;
        points[2 * i + 1] += h;
        grad[2 * i + 1] = (objectif(points, nb, DECOUPAGE) - y) / h;
        points[2 * i + 1] -= h;
    }

    return grad;
    
}

void descente(double* points, int nb, double alpha, double epsilon, double max_iter) {
    double decrement = (alpha-alpha/100)/max_iter;
    for (int i=0; i<max_iter; i++) {
        double* grad = gradient(points, nb);
        for (int j=0; j<nb; j++) {
            points[2 * j] -= alpha * grad[2 * j];
            points[2 * j + 1] -= alpha * grad[2 * j + 1];
        }
        double norme = 0;
        for (int j=0; j<nb; j++) {
            norme += grad[2 * j] * grad[2 * j] + grad[2 * j + 1] * grad[2 * j + 1];
        }
        free(grad);
        if (norme < epsilon*epsilon) {
            break;
        }
        alpha -= decrement;
    }
}

int main(int argc, char const* argv[]) {

    int nb = 1;
    if (argc > 1) {
        nb = atoi(argv[1]);
        for (int i=2; i<argc; i++) {
            zone[i-2] = atoi(argv[i]);
        }
    }

    double* points = malloc(2 * nb * sizeof(double));
    for (int i = 0; i < nb; i++) {
        points[2 * i] = zone[0] + (zone[1] - zone[0]) * rand() / RAND_MAX;
        points[2 * i + 1] = zone[2] + (zone[3] - zone[2]) * rand() / RAND_MAX;
    }

    descente(points, nb, 1, 0.000001, MAX_ITER);

    for (int i = 0; i < 2*nb; i++) {
        printf("%f ", points[i]);
    }

    return EXIT_SUCCESS;

}