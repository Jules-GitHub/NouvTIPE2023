#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int primes[] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29 };

void calcul_base(int dim, int *base) {
    for (int i = 0; i < dim; i++) {
        int p = primes[i];
        int b = 1;
        int n = 1;
        while (n < p) {
            n *= 2;
            b++;
        }
        base[i] = b;
    }
}

void generer_points(int n, int dim, double **points) {
    int base[dim];
    calcul_base(dim, base);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < dim; j++) {
            int b = base[j];
            double xj = 0;
            int d = i;
            while (d > 0) {
                xj += (double)(d % b) / (double)b;
                d /= b;
            }
            points[i][j] = xj;
        }
    }
}

int main() {
    int n = 10;
    int dim = 2;
    double **points = malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        points[i] = malloc(dim * sizeof(double));
    }
    generer_points(n, dim, points);
    for (int i = 0; i < n; i++) {
        printf("(");
        for (int j = 0; j < dim; j++) {
            printf("%.3lf", points[i][j]);
            if (j < dim - 1) {
                printf(", ");
            }
        }
        printf(")\n");
    }
    for (int i = 0; i < n; i++) {
        free(points[i]);
    }
    free(points);
    return EXIT_SUCCESS;
}