import java.util.Scanner;

public class LinearRegression {
    public static void main(String[] args) {
        LinearRegression lr = new LinearRegression();
        Matrixes m = lr.new Matrixes();

        //read the input X and Y
        m.readData();
        //print input matrix or fetch input matrix
        double[][] X = m.printInputMatrix();
        //Compute X transpose
        double[][] Xt = m.transpose(m.X);
        //Compute XtX
        double[][] XtX = m.multiplyxtx(Xt);
        //Compute XtY
        double[] XtY = m.multiplyXtY(Xt);
//        //calculate the cofactor matrix
//        double[][] cofactor = m.cofactorMatrix(XtX, XtX.length);
//        //Calculate the adjugate(transpose of cofactor matrix )
//        double[][] cofactorT = m.transpose(cofactor);
//        //calculate the inverse
//        double[][] inverse = m.inverse(cofactorT);
//       calculate the inverse using gauss jordan elimination as we have more than 5 samples
        double[][] inverse= m.gaussJordanInverse(XtX);
//        multiplying the inverse with XtY to give coefficients
        double[] finalVector = m.multiplyInverseWithXtY(inverse, XtY);
        double [][] newX= m.readTraningSamplesWithoutOutput();
         m.printExamplesMatrix();
        double[] result= m.predictValues(newX,finalVector);

    }

    public class Matrixes {
        //Fetching the feature values by using separate f1 and f2 variables
        //Write the code when you are inputting the values of N and F later
        Scanner sc = new Scanner(System.in);
        int F;
        int N;

        double[][] X;
        double[] Y;


        int T;
        double[][] newX;

        public Matrixes() {
            System.out.println("Enter number of feature");
            F = sc.nextInt();
            System.out.println("Enter number of training records");
            N = sc.nextInt();
            //create the matrices for features X and rows/output N. The reason being you will get exit code 131
            //Read the data
            X = new double[N][F];
            Y = new double[N];
            System.out.println("Enter number of new feature records");
            T = sc.nextInt();
            newX = new double[T][F];
               // System.out.println();
            }

        public void readData() {
            System.out.println("Enter X and Y values:");
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < F; j++) {
                    X[i][j] = sc.nextDouble();
                }
                Y[i] = sc.nextDouble();
            }
        }

        public double[][] readTraningSamplesWithoutOutput() {
            System.out.println("Enter new feature values:");
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < F; j++) {
                    newX[i][j] = sc.nextDouble();
                }
            }
            return newX;
        }

        public double[][] printExamplesMatrix() {
       //print the matrix. There is already a way to direct print array or matrix.You don't have to directly convert the table into array and then print
            System.out.println("new feature Matrix (newX):");
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < F; j++) {
                    System.out.print(newX[i][j] + " ");
                }
                System.out.println();
            }
            return newX;
        }

        public double[][] printInputMatrix() {
//            for (int i = 0; i < N; i++) {
//                for (int j = 0; j < F; j++) {
//                    X[i][j] = sc.nextDouble();
//                }
//            }
            //print the matrix. There is already a way to direct print array or matrix.You don't have to directly convert the table into array and then print
            System.out.println("Feature Matrix (X):");
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < F; j++) {
                    System.out.print(X[i][j] + " ");
                }
                System.out.println();
            }
            return X;
        }

        public double[] printOutputMatrix() {
//            for (int i = 0; i < N; i++) {
//                Y[i] = sc.nextDouble();
//            }
            System.out.println("Output Values (Y):");
            for (int i = 0; i < N; i++) {
                System.out.println(Y[i]);
            }
            return Y;
        }

        //  public class MatrixTranspose{
        public double[][] transpose(double[][] A) {
            int rows = A.length;
            int cols = A[0].length;
            double[][] Xt = new double[cols][rows]; //since this is a transpose (swapped dimensions)
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    Xt[j][i] = A[i][j];
                }
            }
            return Xt;
        }

        // }
        //  public class XtMultiplyX{
        public double[][] multiplyxtx(double[][] Xt) {
            //Find the dimensions
            int rowsX = X.length;
            int colsX = X[0].length;
            System.out.println(colsX);
            int rowsXt = Xt.length;
            System.out.println(rowsXt);
            int colsXt = Xt[0].length;
            //Check if multiplication is possible. It is possible only when colsA=rowsB
            if (colsXt != rowsX) {
                System.out.println("Matrix multiplication is not possible");
                return null;
            }
            //Creating the result matrix Xm
            double[][] Xm = new double[rowsXt][colsX];
            //Multiply the matrices
            for (int i = 0; i < rowsXt; i++) {
                for (int j = 0; j < colsX; j++) {
                    //initialize the cell
                    Xm[i][j] = 0;
                    for (int k = 0; k < colsXt; k++) {
                        Xm[i][j] += Xt[i][k] * X[k][j];
                    }
                }
            }
            // Printing the result
            System.out.println("Result Matrix (Xm):");
            for (int i = 0; i < rowsXt; i++) {
                for (int j = 0; j < colsX; j++) {
                    System.out.print(Xm[i][j] + " ");
                }
                System.out.println();
            }
            return Xm;
        }

        //  public class XtMultiplyY{
        public double[] multiplyXtY(double[][] Xt) {
            int rowsXt = Xt.length;
            int colsXt = Xt[0].length;
            System.out.println(colsXt);
            int rowsY = Y.length;
            System.out.println(rowsY);
            if (colsXt != rowsY) {
                System.out.println("Output Matrix multiplication is not possible");
                return null;
            }
            //creating the result matrix XtYo
            double[] XtYo = new double[rowsXt]; //because Y is a nx1 matrix
            for (int i = 0; i < rowsXt; i++) {
                XtYo[i] = 0;//initialize the first cell
                for (int j = 0; j < colsXt; j++) {
                    XtYo[i] += Xt[i][j] * Y[j];
                }
            }
            for (double v : XtYo) {
                System.out.printf("%.2f", v);
            }
            return XtYo;
        }

        public double determinant(double[][] XXt, int n) {
            //components while calculating determinant-sign(-1 raise to i+j), element of i+j cell,
            //minor matrix obtained after removing i row  and j column
            if (n == 1) {
                return XXt[0][0];
            }
            // int rowsX = XXt.length;
            double determinant = 0;
            //From where that value of n is coming?
            double[][] temp = new double[n - 1][n - 1];
            //n is the value of rows and columns of the matrix of which we are finding the determinant
            int sign = 1;
            for (int i = 0; i < n; i++) { //this is column iteration
                //create/update the cofactor. Minor is created always and only for the first row
                getcofactor(XXt, temp, 0, i, n);
                determinant += sign * XXt[0][i] * determinant(temp, n - 1); //the loop runs for each column in the first row
                sign = -sign;
            }
            return determinant;
        }

      public void getcofactor(double[][] XXt, double[][] temp, int p, int q, int n) {
            int i = 0;
            int j = 0;
            // int rows = XXt.length;
            // int cols = XXt[0].length;
            for (int row = 0; row < n; row++) {
                for (int col = 0; col < n; col++) {
                    if (row != p && col != q) {
                        temp[i][j++] = XXt[row][col];
                        if (j == n - 1) {
                            j = 0;
                            i++;
                        }
                    }
                }
            }
            //cofactor means calculating the multiplication value by deleting the respective rows and columns
            //write these functions on a piece of paper an understand recursion
        }

        double[][] cofactorMatrix(double[][] XXt, int n) {
            double[][] cofactor = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double[][] temp = new double[n - 1][n - 1];
                    getcofactor(XXt, temp, i, j, n);
                    cofactor[i][j] = Math.pow(-1, i + j) * determinant(temp, n - 1);
                }
            }
            return cofactor;
        }

        public double[][] inverse(double[][] cofactorT) {
            int rows = cofactorT.length;
            //  int cols = cofactorT[0].length;
            double determinant = determinant(cofactorT, rows);
            if (determinant == 0) {
                System.out.println("Inverse Matrix is not possible");
                return null;
            }
            double[][] Inverse = new double[rows][rows];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < rows; j++) {
                    Inverse[i][j] = cofactorT[i][j] / determinant;
                }
            }
            return Inverse;
        }

        public double[] multiplyInverseWithXtY(double[][] inverse, double[] outVector) {
            int rows = inverse.length;
            int cols = outVector.length;
            double[] result = new double[rows];
            for (int i = 0; i < rows; i++) {
                result[i]=0;
                for (int j = 0; j < cols; j++) {
                    result[i]+= inverse[i][j] * outVector[j];
                }
            }
            System.out.println("Coeffecients");
            for(int i = 0; i < result.length; i++) {
                System.out.println("B"+i+" = "+result[i]);
            }
            return result;
        }

        // To compute inverse of a square matrix using Gauss Jordan elimination
        public double[][] gaussJordanInverse(double[][] X) {
            int rows = X.length;
            //create augmented matrix [X | I]
            double[][] augmented = new double[rows][2 * rows];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < rows; j++) {
                    augmented[i][j] = X[i][j];
                }
                for (int j =rows; j < 2 * rows; j++) {
                    augmented[i][j] = (i==(j-rows))? 1:0;
                }
            }
            //perform gauss jordan elimination
            for (int i = 0; i < rows; i++) {
                //find the pivot(diagonal element should not be 0)
                double pivot = augmented[i][i];
                if(pivot==0){
                    //find a non-zero row below and swap
                    for (int j = i+1; j < rows; j++) {
                        if(augmented[j][i]!=0){
                            double[] temp= augmented[i];
                            augmented[i]=augmented[j];
                            augmented[j]=temp;
                            pivot=augmented[i][i];
                            break;
                        }
                    }
                }
                if (pivot==0){
                    throw new ArithmeticException("Matrix is singular and cannot be inverted");
                }
                //Normalizing the pivot row
                for (int j = 0; j < 2*rows; j++) {
                    augmented[i][j] /= pivot;
                }
                //Eliminate other rows(everything in the pivot column as 0 except the pivot)
                for (int j = 0; j < rows; j++) {
                    if (j!=i){
                        double factor= augmented[j][i];
                        for (int k = 0; k < 2*rows; k++) {
                            augmented[j][k] -= factor * augmented[i][k];
                        }
                    }
                }
            }
            //extract right half -> inverse matrix
            double[][]gaussJordanInverse=new double[rows][rows];
            for(int i=0; i<rows; i++)
            {
                for (int j = 0; j < rows; j++){
                    gaussJordanInverse[i][j]=augmented[i][j+rows];
                }
            }
            for (double[] row : gaussJordanInverse) {
                for (double value : row) {
                    System.out.printf("%10.4f",value);
                }
                System.out.println();
            }
            return gaussJordanInverse;
        }

        public double[] predictValues(double[][] features, double[] coefficients) {
            double[] result= new double[features.length];
            for (int i = 0; i < features.length; i++) {
                for (int j = 0; j < features[0].length; j++) {
                    result[i] += features[i][j] * coefficients[j];
                }
            }
            System.out.println("Predicted Values");
            for(int i = 0; i < result.length; i++) {
                System.out.printf("%10.4f",result[i]);
                System.out.println();
            }
            return result;
        }
    }


}
