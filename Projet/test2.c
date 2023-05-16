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

double* essaim_particulaire(double** particules, int nbPoints, int nbParticules, int nbIterations, int decoupage) {

    time_t t0 = time(0);

    double** vitesse = malloc(nbParticules * sizeof(double*));
    for (int i = 0; i < nbParticules; i++) {
        vitesse[i] = malloc(2 * nbPoints * sizeof(double));
        for (int j = 0; j < 2 * nbPoints; j++) {
            vitesse[i][j] = 0;
        }
    }

    double* pBest = malloc(nbParticules * sizeof(double));
    double** pBestPoints = malloc(nbParticules * sizeof(double*));
    for (int i = 0; i < nbParticules; i++) {
        pBestPoints[i] = malloc(2 * nbPoints * sizeof(double));
    }

    double gBest = INFINITY;
    double* gBestPoints = malloc(2 * nbPoints * sizeof(double));

    for (int i = 0; i < nbParticules; i++) {
        pBest[i] = objectif(particules[i], nbPoints, decoupage);
        for (int j = 0; j < 2 * nbPoints; j++) {
            pBestPoints[i][j] = particules[i][j];
        }
        if (pBest[i] < gBest) {
            gBest = pBest[i];
            for (int j = 0; j < 2 * nbPoints; j++) {
                gBestPoints[j] = particules[i][j];
            }
        }
    }

    double w = 0.729;
    double c1 = 1.49445;
    double c2 = 1.49445;

    for (int n=0; n<nbIterations; n++) {

        for (int i = 0; i < nbParticules; i++) {

            for (int j = 0; j < 2 * nbPoints; j++) {
                vitesse[i][j] = w * vitesse[i][j] + c1 * rand() / RAND_MAX * (pBestPoints[i][j] - particules[i][j]) + c2 * rand() / RAND_MAX * (gBestPoints[j] - particules[i][j]);
                particules[i][j] += vitesse[i][j];
            }

            double o = objectif(particules[i], nbPoints, decoupage);
            if (o < pBest[i]) {
                pBest[i] = o;
                for (int j = 0; j < 2 * nbPoints; j++) {
                    pBestPoints[i][j] = particules[i][j];
                }
                if (o < gBest) {
                    gBest = o;
                    for (int j = 0; j < 2 * nbPoints; j++) {
                        gBestPoints[j] = particules[i][j];
                    }
                }
            }

        }

        printf("Itération n°%d, durée = %ld, gBest = %f\r", n, time(0)-t0, gBest);
        t0 = time(0);

    }

    return gBestPoints;

}


int main() {

    srand(time(0));

    time_t tDebut = time(0);

    int nbPoints = 10;
    int nbParticules = 2*nbPoints;

    double** particules = malloc(nbParticules * sizeof(double*));
    for (int i = 0; i < nbParticules; i++) {
        particules[i] = malloc(2 * nbPoints * sizeof(double));
        for (int j=0; j<nbPoints; j++) {
            particules[i][2*j] = zone[0] + (zone[1]-zone[0]) * rand() / RAND_MAX;
            particules[i][2*j+1] = zone[2] + (zone[3]-zone[2]) * rand() / RAND_MAX;
        }
    }

    double* points = essaim_particulaire(particules, nbPoints, nbParticules, MAX_ITER, DECOUPAGE);

    printf("\n");
    printf("Temps total : %ld\n", time(0) - tDebut);
    printf("Objectif = %f\n", objectif(points, nbPoints, 10*DECOUPAGE));

    for (int i = 0; i < nbPoints; i++) {
        printf("%f, %f,\n", points[2 * i], points[2 * i + 1]);
    }

    return EXIT_SUCCESS;

}