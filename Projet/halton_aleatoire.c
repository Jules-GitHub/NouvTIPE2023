#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double halton(int i, int base) {
    double f = 1;
    double r = 0;
    while (i > 0) {
        f /= base;
        r += f * (i % base);
        i /= base;
    }
    return r;
}

double halton_random(int i, int base, double alpha) {
    double h = halton(i, base);
    double u = (double) rand() / RAND_MAX;
    return h + alpha * u;
}

int main() {
    int n = 10; // nombre de points à générer
    int dim = 2; // dimension de l'espace
    int* bases = malloc(dim * sizeof(int)); // tableau des bases
    double alpha = 0.1; // paramètre de perturbation aléatoire
    int i, j;

    // Initialisation du générateur de nombres aléatoires
    srand(time(NULL));

    // Initialisation des bases
    for (i = 0; i < dim; i++) {
        bases[i] = i + 2; // choix des bases
    }

    // Génération des points
    double** points = malloc(n * sizeof(double*));
    for (i = 0; i < n; i++) {
        points[i] = malloc(dim * sizeof(double));
        for (j = 0; j < dim; j++) {
            points[i][j] = halton_random(i, bases[j], alpha);
        }
    }

    // Affichage des points
    for (i = 0; i < n; i++) {
        printf("Point %d: (", i);
        for (j = 0; j < dim; j++) {
            printf("%f", points[i][j]);
            if (j < dim - 1) {
                printf(", ");
            }
        }
        printf(")\n");
    }

    // Libération de la mémoire allouée
    for (i = 0; i < n; i++) {
        free(points[i]);
    }
    free(points);
    free(bases);

    return 0;
}