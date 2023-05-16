#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "time.h"

double zone[4] = {0, 1, 0, 1};
int DECOUPAGE = 100;
int MAX_ITER = 100000;

struct attracteur_s {
    double x;
    double y;
    double coeff;
};
typedef struct attracteur_s attracteur;

double max(double a, double b) {
    return a<b ? b : a;
}

double coeff_point(double x, double y, attracteur* attracteurs, int nbAtt) {

    double* distAtt = malloc(nbAtt*sizeof(attracteur));
    for (int i=0; i<nbAtt; i++) {
        double dx = attracteurs[i].x - x;
        double dy = attracteurs[i].y - y;
        double d = sqrt(dx * dx + dy * dy);
        distAtt[i] = d;
    }

    double coeff = 0;
    double somme = 0;
    for (int i=0; i<nbAtt; i++) {
        somme += 1/(1+distAtt[i]);
        coeff += attracteurs[i].coeff/(1+distAtt[i]);
    }

    return coeff/somme;

}

double coeff_point2(double x, double y, attracteur* attracteurs, int nbAtt) {

    double dist = INFINITY;
    double coeff = 0;

    for (int i=0; i<nbAtt; i++) {
        double dx = attracteurs[i].x - x;
        double dy = attracteurs[i].y - y;
        double d = sqrt(dx * dx + dy * dy);
        if (d < dist) {
            dist = d;
            coeff = attracteurs[i].coeff*exp(-1/pow(sqrt(2)-dist, 5));
        }
    }
    
    return coeff;

}

double coeff_point3(double x, double y, attracteur* attracteurs, int nbAtt) {

    double dist = INFINITY;
    double coeff = 0;

    for (int i=0; i<nbAtt; i++) {
        double dx = attracteurs[i].x - x;
        double dy = attracteurs[i].y - y;
        double d = sqrt(dx * dx + dy * dy);
        if (d < dist) {
            dist = d;
            coeff = attracteurs[i].coeff/(1+pow(dist, 5));
        }
    }
    
    return coeff;

}

double** tab_coeff(attracteur* attracteurs, int nbAtt, int decoupage) {

    double** tab = malloc((decoupage+1)*sizeof(double*));
    for (int i=0; i<=decoupage; i++) {
        tab[i] = malloc((decoupage+1)*sizeof(double));
        for (int j=0; j<=decoupage; j++) {
            tab[i][j] = coeff_point2(((double) i)/decoupage, ((double) j)/decoupage, attracteurs, nbAtt);
        }
    }

    return tab;

}

double distance(double* points, int nb, attracteur* attracteurs, int nbAtt, double x, double y, double coeff) {

    double min = INFINITY;

    for (int i = 0; i < nb; i++) {
        double dx = points[2 * i] - x;
        double dy = points[2 * i + 1] - y;
        double d = dx * dx + dy * dy;
        if (d < min) {
            min = d;
        }
    }

    min = sqrt(min)*(1+coeff);

    return min;

}

double simpson(double* points, int nb, attracteur* attracteurs, int nbAtt, double x0, double x1, double y0, double y1, int n, double** coeffs) {

    double h = (x1 - x0) / n;
    double k = (y1 - y0) / n;

    double integral = 0;

    for (int i = 0; i <= n; i++) {

        double x = x0 + i * h;
        double wx = (i == 0 || i == n) ? 1 : (i % 2 == 1) ? 4 : 2;

        for (int j = 0; j <= n; j++) {

            double y = y0 + j * k;
            double wy = (j == 0 || j == n) ? 1 : (j % 2 == 1) ? 4 : 2;

            double d = distance(points, nb, attracteurs, nbAtt, x, y, coeffs[i][j]);

            integral += wx * wy * d;
        }

    }

    integral *= h * k / 9.0;
    return integral;

}

double objectif(double* points, int nb, attracteur* attracteurs, int nbAtt, int decoupage, double** coeffs) {
    return simpson(points, nb, attracteurs, nbAtt, 0, 1, 0, zone[3]-zone[2], decoupage, coeffs)/(zone[3]-zone[2]);
}


int main(int argc, char const* argv[]) {

    srand(time(NULL));

    if (argc < 3) {
        printf("Usage: %s nbPoints nbAttracteurs [x1 y1] [x2 y2] ... [x1 y1 coeff1] [x2 y2 coeff2] ...\n", argv[0]);
        return 1;
    }

    int nb = atoi(argv[1]);
    int nbAtt = atoi(argv[2]);

    double* points = malloc(2 * nb * sizeof(double));
    for (int i = 0; i < 2*nb; i++) {
        points[i] = atof(argv[i + 3]);
    }

    attracteur* attracteurs = malloc(nbAtt * sizeof(attracteur));

    for (int i=0; i<nbAtt; i++) {
        attracteurs[i].x = (atof(argv[3*i + 2*nb+3])-zone[0])/(zone[1]-zone[0]);
        attracteurs[i].y = (atof(argv[3*i + 2*nb+4])-zone[2])/(zone[1]-zone[0]); // Pour garder les rapports entre les longueurs
        attracteurs[i].coeff = atof(argv[3*i + 2*nb+5]);
    }

    double** coeffs = tab_coeff(attracteurs, nbAtt, DECOUPAGE);

    printf("%f", objectif(points, nb, attracteurs, nbAtt, DECOUPAGE, coeffs));

    free(points);
    free(attracteurs);
    
    for (int i=0; i<=DECOUPAGE; i++) {
        free(coeffs[i]);
    }
    free(coeffs);

    return EXIT_SUCCESS;

}