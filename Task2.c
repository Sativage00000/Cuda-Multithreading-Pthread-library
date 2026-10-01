/*
 Author: Oladipupo Roland Familua
 Student Number: 2326310
 Course: BSc Computer Science
 Module: 6CS005
 Description:
 This program performs parallel matrix operations using OpenMP.
 It reads pairs of matrices from a file, applies arithmetic and
 matrix transformations, and writes the results to an output file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* Matrix structure definition */
typedef struct {
    int rows;
    int columns;
    double **data;
} Matrix;

/*
 Reads a matrix from the input file.
 Expected format:
 rows,columns
 v1,v2,v3,...
*/
Matrix readMatrix(FILE *file) {
    Matrix matrix;
    matrix.rows = 0;
    matrix.columns = 0;
    matrix.data = NULL;

    /* Read matrix dimensions */
    if (fscanf(file, "%d,%d", &matrix.rows, &matrix.columns) != 2) {
    if (!feof(file)) {
        fprintf(stderr, "Failed to read matrix dimensions.\n");
    }
    matrix.rows = 0;
    matrix.columns = 0;
    matrix.data = NULL;
    return matrix;
}
    /* Allocate memory */
    matrix.data = (double **) malloc(matrix.rows * sizeof(double *));
    for (int i = 0; i < matrix.rows; i++) {
        matrix.data[i] = (double *) malloc(matrix.columns * sizeof(double));

        for (int j = 0; j < matrix.columns; j++) {
            if (fscanf(file, "%lf", &matrix.data[i][j]) != 1) {

                /* Cleanup on failure */
                for (int k = 0; k < i; k++) {
                    free(matrix.data[k]);
                }
                free(matrix.data);

                matrix.rows = 0;
                matrix.columns = 0;
                matrix.data = NULL;

                fprintf(stderr, "Failed to read matrix data.\n");
                return matrix;
            }

            /* Consume delimiter */
            fgetc(file);
        }
    }

    return matrix;
}

/* Frees allocated matrix memory */
void freeMatrix(Matrix matrix) {
    if (matrix.data) {
        for (int i = 0; i < matrix.rows; i++) {
            free(matrix.data[i]);
        }
        free(matrix.data);
    }
}

/* Creates an empty matrix of given size */
Matrix createMatrix(int rows, int columns) {
    Matrix matrix;
    matrix.rows = rows;
    matrix.columns = columns;
    matrix.data = (double **) malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) {
        matrix.data[i] = (double *) malloc(columns * sizeof(double));
    }
    return matrix;
}

/* Element-wise matrix addition */
Matrix addMatrices(Matrix A, Matrix B, int threads) {
    Matrix result = createMatrix(A.rows, A.columns);

    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.columns; j++) {
            result.data[i][j] = A.data[i][j] + B.data[i][j];
        }
    }
    return result;
}

/* Element-wise matrix subtraction */
Matrix subtractMatrices(Matrix A, Matrix B, int threads) {
    Matrix result = createMatrix(A.rows, A.columns);

    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.columns; j++) {
            result.data[i][j] = A.data[i][j] - B.data[i][j];
        }
    }
    return result;
}

/* Element-wise matrix multiplication */
Matrix multiplyMatrices(Matrix A, Matrix B, int threads) {
    Matrix result = createMatrix(A.rows, A.columns);

    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.columns; j++) {
            result.data[i][j] = A.data[i][j] * B.data[i][j];
        }
    }
    return result;
}

/* Element-wise matrix division */
Matrix divideMatrices(Matrix A, Matrix B, int threads) {
    Matrix result = createMatrix(A.rows, A.columns);

    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.columns; j++) {
            if (B.data[i][j] == 0.0) {
                result.data[i][j] = 0.0 / 0.0; /* NaN */
            } else {
                result.data[i][j] = A.data[i][j] / B.data[i][j];
            }
        }
    }
    return result;
}

/* Matrix transpose */
Matrix transposeMatrix(Matrix matrix, int threads) {
    Matrix result = createMatrix(matrix.columns, matrix.rows);

    int maxThreads = threads;
    if (matrix.rows < threads) {
        maxThreads = matrix.rows;
    }

    #pragma omp parallel for num_threads(maxThreads)
    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.columns; j++) {
            result.data[j][i] = matrix.data[i][j];
        }
    }
    return result;
}

/* Standard matrix multiplication */
Matrix multplyVectorMatrices(Matrix A, Matrix B, int threads) {
    Matrix result = createMatrix(A.rows, B.columns);

    int maxThreads = threads;
    if (A.rows < threads) {
        maxThreads = A.rows;
    }

    #pragma omp parallel for num_threads(maxThreads)
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < B.columns; j++) {
            result.data[i][j] = 0.0;
            for (int k = 0; k < A.columns; k++) {
                result.data[i][j] += A.data[i][k] * B.data[k][j];
            }
        }
    }
    return result;
}

/* Saves a matrix to the output file */
void saveMatrixToFile(FILE *file, char *operationName, Matrix matrix) {
    fprintf(file, "%s - %d,%d\n", operationName, matrix.rows, matrix.columns);

    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.columns; j++) {
            fprintf(file, "%.6f", matrix.data[i][j]);
            if (j < matrix.columns - 1) {
                fprintf(file, ",");
            }
        }
        fprintf(file, "\n");
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <filename> <num_threads>\n", argv[0]);
        return 1;
    }

    FILE *file = fopen(argv[1], "r");
    if (!file) {
        printf("Unable to open input file.\n");
        return 1;
    }

    FILE *outputFile = fopen("results.txt", "w");
    if (!outputFile) {
        printf("Unable to open output file.\n");
        return 1;
    }

    int threads = atoi(argv[2]);
    int pairCount = 0;

    while (1) {
        Matrix A = readMatrix(file);
        if (A.rows == 0) break;

        Matrix B = readMatrix(file);
        if (B.rows == 0) {
            freeMatrix(A);
            break;
        }

        fprintf(outputFile, "\n** MATRIX PAIR NUMBER: %d **\n", pairCount);

        if (A.rows == B.rows && A.columns == B.columns) {
            int maxThreads = (A.rows < threads) ? A.rows : threads;

            Matrix add = addMatrices(A, B, maxThreads);
            saveMatrixToFile(outputFile, "Addition", add);
            freeMatrix(add);

            Matrix sub = subtractMatrices(A, B, maxThreads);
            saveMatrixToFile(outputFile, "Subtraction", sub);
            freeMatrix(sub);

            Matrix mul = multiplyMatrices(A, B, maxThreads);
            saveMatrixToFile(outputFile, "Element-by-element multiplication", mul);
            freeMatrix(mul);

            Matrix div = divideMatrices(A, B, maxThreads);
            saveMatrixToFile(outputFile, "Element-by-element division", div);
            freeMatrix(div);
        } else {
            fprintf(outputFile, "Element-wise operations cannot be performed (shape mismatch).\n");
        }

        Matrix tA = transposeMatrix(A, threads);
        saveMatrixToFile(outputFile, "Transposed Matrix A", tA);
        freeMatrix(tA);

        Matrix tB = transposeMatrix(B, threads);
        saveMatrixToFile(outputFile, "Transposed Matrix B", tB);
        freeMatrix(tB);

        if (A.columns == B.rows) {
            Matrix mm = multplyVectorMatrices(A, B, threads);
            saveMatrixToFile(outputFile, "Matrix multiplication", mm);
            freeMatrix(mm);
        } else {
            fprintf(outputFile, "Matrix multiplication cannot be performed (shape mismatch).\n");
        }

        freeMatrix(A);
        freeMatrix(B);
        pairCount++;
    }

    fclose(file);
    fclose(outputFile);
    return 0;
}
