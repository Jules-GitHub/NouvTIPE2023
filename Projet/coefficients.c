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

    int iMin = 0;

    for (int i=0; i<nbAtt; i++) {
        double dx = attracteurs[i].x - x;
        double dy = attracteurs[i].y - y;
        double d = sqrt(dx * dx + dy * dy);
        if (d < dist) {
            iMin = i;
            dist = d;
            coeff = attracteurs[i].coeff*exp(-1/pow(sqrt(2)-dist, 5));
        }
    }

    /*if (iMin == 10) printf("Ouiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii\n");
    printf("%f %f %d %f %f\n\n", x, y, iMin, coeff, dist);*/
    
    return coeff;

}

double coeff_point3(double x, double y, attracteur* attracteurs, int nbAtt) {

    double dist = INFINITY;
    double coeff = 0;

    for (int i=0; i<nbAtt; i++) {
        double dx = attracteurs[i].x - x;
        double dy = attracteurs[i].y - y;
        double d = dx * dx + dy * dy;
        if (d < dist) {
            dist = d;
            coeff = attracteurs[i].coeff/(1+pow(dist, 1000));
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

int main(int argc, char const* argv[]) {

    srand(time(NULL));

    if (argc < 3) {
        printf("Usage: %s nbAttracteurs [x1 y1 coeff1] [x2 y2 coeff2] ...\n", argv[0]);
        return 1;
    }

    int nbAtt = atoi(argv[1]);

    attracteur* attracteurs = malloc(nbAtt * sizeof(attracteur));

    for (int i=0; i<nbAtt; i++) {
        attracteurs[i].x = (atof(argv[3*i+2])-zone[0])/(zone[1]-zone[0]);
        attracteurs[i].y = (atof(argv[3*i+3])-zone[2])/(zone[1]-zone[0]); // Pour garder les rapports entre les longueurs
        attracteurs[i].coeff = atof(argv[3*i+4]);
    }

    double** coeffs = tab_coeff(attracteurs, nbAtt, DECOUPAGE);

    for (int i=0; i<=DECOUPAGE; i++) {
        for (int j=0; j<=DECOUPAGE; j++) {
            printf("%f ", coeffs[i][j]);
        }
    }

    free(attracteurs);
    
    for (int i=0; i<=DECOUPAGE; i++) {
        free(coeffs[i]);
    }
    free(coeffs);

    return EXIT_SUCCESS;

}