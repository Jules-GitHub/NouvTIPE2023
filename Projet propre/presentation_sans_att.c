#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "time.h"

int DECOUPAGE = 100;

/* Partie sans attracteurs */
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

double integrale(double* points, int nb, double x0, double x1, double y0, double y1, int n) {
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
    return integrale(points, nb, 0, 1, 0, 1, decoupage);
}

void gradient(double* points, int nb, double* grad) {
    double h = 0.00001;
    double y = objectif(points, nb, DECOUPAGE);
    for (int i = 0; i < 2*nb; i++) {
        points[i] += h;
        grad[i] = (objectif(points, nb, DECOUPAGE) - y) / h;
        points[i] -= h;
    }
}

void descente(double* points, int nb, double alpha, double epsilon, double max_iter) {
    double decrement = (alpha-alpha/100)/max_iter;
    double* grad = malloc(2*nb*sizeof(double));
    for (int i=0; i<max_iter; i++) {
        gradient(points, nb, grad);
        for (int j=0; j<2*nb; j++) {
            points[j] -= alpha * grad[j];
        }
        double norme = 0;
        for (int j=0; j<nb; j++) {
            norme += grad[2*j]*grad[2*j] + grad[2*j+1]*grad[2*j+1];
        }
        if (norme < epsilon*epsilon) break;
        alpha -= decrement;
    }
    free(grad);
}

int main() {

    return EXIT_SUCCESS;

}