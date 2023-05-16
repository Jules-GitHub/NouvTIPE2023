#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "time.h"

double zone[4] = {0, 4, 0, 4};
int DECOUPAGE = 100;
int MAX_ITER = 100000;

struct attracteur_s {
    double x;
    double y;
    double coeff;
};
typedef struct attracteur_s attracteur;

double max(double a, double b) {
    return a > b ? a : b;
}

double distance(double* points, int nb, attracteur* attracteurs, int nbAtt, double x, double y) {

    double min = INFINITY;

    for (int i = 0; i < nb; i++) {
        double dx = points[2 * i] - x;
        double dy = points[2 * i + 1] - y;
        double d = dx * dx + dy * dy;
        if (d < min) {
            min = d;
        }
    }

    min = sqrt(min);

    for (int i = 0; i < nbAtt; i++) {
        double dx = attracteurs[i].x - x;
        double dy = attracteurs[i].y - y;
        double d = dx * dx + dy * dy;
        min *= 1 + attracteurs[i].coeff/(1+d);
    }

    return min;

}

double simpson(double* points, int nb, attracteur* attracteurs, int nbAtt, double x0, double x1, double y0, double y1, int n) {

    double h = (x1 - x0) / n;
    double k = (y1 - y0) / n;

    double integral = 0;

    for (int i = 0; i <= n; i++) {

        double x = x0 + i * h;
        double wx = (i == 0 || i == n) ? 1 : (i % 2 == 1) ? 4 : 2;

        for (int j = 0; j <= n; j++) {

            double y = y0 + j * k;
            double wy = (j == 0 || j == n) ? 1 : (j % 2 == 1) ? 4 : 2;

            integral += wx * wy * distance(points, nb, attracteurs, nbAtt, x, y);
        }

    }

    integral *= h * k / 9.0;
    return integral;

}

double objectif(double* points, int nb, attracteur* attracteurs, int nbAtt, int decoupage) {
    return simpson(points, nb, attracteurs, nbAtt, zone[0], zone[1], zone[2], zone[3], decoupage)/(zone[1]-zone[0])/(zone[3]-zone[2]);
}

void gradient(double* points, int nb, attracteur* attracteurs, int nbAtt, double* grad) {
    
    double h = 0.00001;
    double y = objectif(points, nb, attracteurs, nbAtt, DECOUPAGE);
    for (int i = 0; i < nb; i++) {
        points[2 * i] += h;
        grad[2 * i] = (objectif(points, nb, attracteurs, nbAtt, DECOUPAGE) - y) / h;
        points[2 * i] -= h;
        points[2 * i + 1] += h;
        grad[2 * i + 1] = (objectif(points, nb, attracteurs, nbAtt, DECOUPAGE) - y) / h;
        points[2 * i + 1] -= h;
    }
    
}

void descente(double* points, int nb, attracteur* attracteurs, int nbAtt, double alpha, double epsilon, double max_iter) {
    time_t t0 = time(NULL);
    double decrement = (alpha-alpha/100)/max_iter;
    double* grad = malloc(2*nb*sizeof(double));
    for (int i=0; i<max_iter; i++) {
        gradient(points, nb, attracteurs, nbAtt, grad);
        printf("Gradient : \n");
        for (int j=0; j<nb; j++) {
            printf("%f %f\n", grad[2*j], grad[2*j+1]);
        }
        printf("Objectif : %f\n", objectif(points, nb, attracteurs, nbAtt, DECOUPAGE));
        printf("Points :\n");
        for (int j=0; j<nb; j++) {
            printf("%f %f\n", points[2*j], points[2*j+1]);
        }
        for (int j=0; j<nb; j++) {
            points[2 * j] -= alpha * grad[2 * j];
            points[2 * j + 1] -= alpha * grad[2 * j + 1];
        }
        double norme = 0;
        for (int j=0; j<nb; j++) {
            norme += grad[2 * j] * grad[2 * j] + grad[2 * j + 1] * grad[2 * j + 1];
        }
        if (norme < epsilon*epsilon) {
            break;
        }
        alpha -= decrement;
        printf("Itération n°%d, durée = %ld, ||grad||2 = %f, alpha = %f\n", i, time(NULL)-t0, norme, alpha);
        t0 = time(NULL);
    }
    free(grad);
}

int main() {

    srand(time(NULL));

    int nb = 5;

    int nbAtt = 20;
    attracteur* attracteurs = malloc(nbAtt * sizeof(attracteur));
    for (int i=0; i<nbAtt; i+=3) {
        attracteurs[i].x = zone[0] + (zone[1] - zone[0])*rand()/RAND_MAX;
        attracteurs[i].y = zone[2] + (zone[3] - zone[2])*rand()/RAND_MAX;
        attracteurs[i].coeff = 10*(1+rand()/RAND_MAX);
    }

    double* points = malloc(2 * nb * sizeof(double));
    for (int i = 0; i < nb; i++) {
        points[2 * i] = zone[0] + (zone[1] - zone[0]) * rand() / RAND_MAX;
        points[2 * i + 1] = zone[2] + (zone[3] - zone[2]) * rand() / RAND_MAX;
    }

    double alpha = 1/(objectif(points, nb, attracteurs, nbAtt, DECOUPAGE));

    descente(points, nb, attracteurs, nbAtt, alpha, 0.000001, MAX_ITER);

    for (int i = 0; i < 2*nb; i++) {
        printf("%f ", points[i]);
    }

    return EXIT_SUCCESS;

}