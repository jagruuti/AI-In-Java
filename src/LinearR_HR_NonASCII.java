import java.util.Scanner;

public class LinearR_HR_NonASCII {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        Matrixes m = new Matrixes(sc);

        // Step 1: Read training data
        m.readData();

        // Step 2: Compute intermediate matrices
        double[][] Xt = m.transpose(m.X);
        double[][] XtX = m.multiplyxtx(Xt);
        double[] XtY = m.multiplyXtY(Xt);

        // Step 3: Compute inverse using Gauss-Jordan
        double[][] inverse = m.gaussJordanInverse(XtX);

        // Step 4: Compute coefficients
        double[] finalVector = m.multiplyInverseWithXtY(inverse, XtY);

        // Step 5: Read test data (without Y)
        double[][] newX = m.readTraningSamplesWithoutOutput();

        // Step 6: Predict results
        double[] result = m.predictValues(newX, finalVector);

        // Step 7: Print results
        for (double v : result) {
            System.out.printf("%.2f\n", v);
        }
    }

    public static class Matrixes {
        Scanner sc;
        int F;
        int N;
        int T;

        double[][] X;
        double[] Y;
        double[][] newX;

        public Matrixes(Scanner sc) {
            this.sc = sc;
        }

        public void readData() {
            F = sc.nextInt();
            N = sc.nextInt();
            X = new double[N][F];
            Y = new double[N];

            for (int i = 0; i < N; i++) {
                for (int j = 0; j < F; j++) {
                    X[i][j] = sc.nextDouble();
                }
                Y[i] = sc.nextDouble();
            }

            T = sc.nextInt();
            newX = new double[T][F];
        }

        public double[][] readTraningSamplesWithoutOutput() {
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < F; j++) {
                    newX[i][j] = sc.nextDouble();
                }
            }
            return newX;
        }

        public double[][] transpose(double[][] A) {
            int rows = A.length;
            int cols = A[0].length;
            double[][] Xt = new double[cols][rows];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    Xt[j][i] = A[i][j];
                }
            }
            return Xt;
        }

        public double[][] multiplyxtx(double[][] Xt) {
            int rowsX = X.length;
            int colsX = X[0].length;
            int rowsXt = Xt.length;
            int colsXt = Xt[0].length;
            double[][] Xm = new double[rowsXt][colsX];
            for (int i = 0; i < rowsXt; i++) {
                for (int j = 0; j < colsX; j++) {
                    for (int k = 0; k < colsXt; k++) {
                        Xm[i][j] += Xt[i][k] * X[k][j];
                    }
                }
            }
            return Xm;
        }

        public double[] multiplyXtY(double[][] Xt) {
            int rowsXt = Xt.length;
            int colsXt = Xt[0].length;
            double[] XtYo = new double[rowsXt];
            for (int i = 0; i < rowsXt; i++) {
                for (int j = 0; j < colsXt; j++) {
                    XtYo[i] += Xt[i][j] * Y[j];
                }
            }
            return XtYo;
        }

        public double[][] gaussJordanInverse(double[][] X) {
            int rows = X.length;
            double[][] augmented = new double[rows][2 * rows];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < rows; j++) {
                    augmented[i][j] = X[i][j];
                }
                for (int j = rows; j < 2 * rows; j++) {
                    augmented[i][j] = (i == (j - rows)) ? 1 : 0;
                }
            }

            for (int i = 0; i < rows; i++) {
                double pivot = augmented[i][i];
                if (pivot == 0) {
                    for (int j = i + 1; j < rows; j++) {
                        if (augmented[j][i] != 0) {
                            double[] temp = augmented[i];
                            augmented[i] = augmented[j];
                            augmented[j] = temp;
                            pivot = augmented[i][i];
                            break;
                        }
                    }
                }
                if (pivot == 0) {
                    throw new ArithmeticException("Matrix is singular and cannot be inverted");
                }

                for (int j = 0; j < 2 * rows; j++) {
                    augmented[i][j] /= pivot;
                }

                for (int j = 0; j < rows; j++) {
                    if (j != i) {
                        double factor = augmented[j][i];
                        for (int k = 0; k < 2 * rows; k++) {
                            augmented[j][k] -= factor * augmented[i][k];
                        }
                    }
                }
            }

            double[][] inverse = new double[rows][rows];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < rows; j++) {
                    inverse[i][j] = augmented[i][j + rows];
                }
            }
            return inverse;
        }

        public double[] multiplyInverseWithXtY(double[][] inverse, double[] outVector) {
            int rows = inverse.length;
            int cols = outVector.length;
            double[] result = new double[rows];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result[i] += inverse[i][j] * outVector[j];
                }
            }
            return result;
        }

        public double[] predictValues(double[][] features, double[] coefficients) {
            double[] result = new double[features.length];
            for (int i = 0; i < features.length; i++) {
                for (int j = 0; j < features[0].length; j++) {
                    result[i] += features[i][j] * coefficients[j];
                }
            }
            return result;
        }
    }
}