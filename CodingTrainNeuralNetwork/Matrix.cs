using System;
using System.Collections.Generic;
using System.Security.Cryptography;

namespace CodingTrainNeuralNetwork {

    /// <summary>
    /// A helper class for the linear algebra stuff needed in an NN project
    /// </summary>
    public class Matrix {

        /// <summary>
        /// Gets the number of rows
        /// </summary>
        public int Rows { get; private set; }

        /// <summary>
        /// Gets the number of columns
        /// </summary>
        public int Cols { get; private set; }

        /// <summary>
        /// The matrix data
        /// </summary>
        public readonly double[,] Data;

        /// <summary>
        /// Creates an instance of this Matrix class
        /// </summary>
        public Matrix(int rows, int cols) {
            this.Rows = rows;
            this.Cols = cols;
            this.Data = new double[this.Rows, this.Cols];

            for (int i = 0; i < this.Rows; i++) {
                for (int j = 0; j < this.Cols; j++) {
                    this.Data[i, j] = 0f;
                }
            }
        }

        /// <summary>
        /// Creates a new Matrix from an array
        /// </summary>
        /// <param name="arr"></param>
        /// <returns></returns>
        public static Matrix FromArray(double[] arr) {
            var m = new Matrix(arr.Length, 1);
            for (int i = 0; i < arr.Length; i++) {
                m.Data[i, 0] = arr[i];
            }
            return m;
        }

        /// <summary>
        /// Returns a new Matrx a-b
        /// </summary>
        /// <param name="a">Subtrahend</param>
        /// <param name="b">Subtractor</param>
        /// <returns>Difference of a and b</returns>
        public static Matrix Subtract(Matrix a, Matrix b) {
            var result = new Matrix(a.Rows, a.Cols);
            for (int i = 0; i < result.Rows; i++) {
                for (int j = 0; j < result.Cols; j++) {
                    result.Data[i, j] = a.Data[i, j] - b.Data[i, j];
                }
            }
            return result;
        }

        /// <summary>
        /// Converts this Matrix into a list of doubleing numbers
        /// </summary>
        /// <returns></returns>
        public double[] ToArray() {
            var arr = new List<double>();
            for (int i = 0; i < this.Rows; i++) {
                for (int j = 0; j < this.Cols; j++) {
                    arr.Add(this.Data[i, j]);
                }
            }
            return arr.ToArray();
        }

        /// <summary>
        /// Feeds the data with random numbers
        /// </summary>
        public void Randomize() {
            for (int i = 0; i < this.Rows; i++) {
                for (int j = 0; j < this.Cols; j++) {
                    using (var rng = new RNGCryptoServiceProvider()) {
                        byte[] result = new byte[8];
                        rng.GetBytes(result);
                        this.Data[i, j] = (double)BitConverter.ToUInt64(result, 0) / ulong.MaxValue;
                    }
                }
            }
        }

        /// <summary>
        /// Adds value of Matrix-n into the value of this object
        /// </summary>
        /// <param name="n"></param>
        public void Add(Matrix n) {
            for (int i = 0; i < this.Rows; i++) {
                for (int j = 0; j < this.Cols; j++) {
                    this.Data[i, j] += (n as Matrix).Data[i, j];
                }
            }
        }

        /// <summary>
        /// Adds value of scalar-n into the value of this object
        /// </summary>
        /// <param name="n"></param>
        public void Add(double n) {
            for (int i = 0; i < this.Rows; i++) {
                for (int j = 0; j < this.Cols; j++) {
                    this.Data[i, j] += n;
                }
            }
        }

        /// <summary>
        /// Transposes a given matrix parameter
        /// </summary>
        /// <param name="m"></param>
        public static Matrix Transpose(Matrix m) {
            var result = new Matrix(m.Cols, m.Rows);
            for (int i = 0; i < m.Rows; i++) {
                for (int j = 0; j < m.Cols; j++) {
                    result.Data[j, i] = m.Data[i, j];
                }
            }
            return result;
        }

        /// <summary>
        /// Returns a new Matrix a * b
        /// </summary>
        /// <param name="a">Multiplicand</param>
        /// <param name="b">multiplier</param>
        /// <returns>The product of a and b</returns>
        public static Matrix Multiply(Matrix a, Matrix b) {
            if (a.Cols != b.Rows) {
                throw new Exception("Columns of A must match rows of B.");
            }

            var result = new Matrix(a.Rows, b.Cols);
            for (int i = 0; i < result.Rows; i++) {
                for (int j = 0; j < result.Cols; j++) {
                    var sum = default(double);
                    for (int k = 0; k < a.Cols; k++) {
                        sum += a.Data[i, k] * b.Data[k, j];
                    }
                    result.Data[i, j] = sum;
                }
            }
            return result;
        }

        /// <summary>
        /// Gets the Hadamard product
        /// </summary>
        /// <param name="n">Multiplier</param>
        /// <returns></returns>
        public Matrix Multiply(Matrix n) {
            var result = new Matrix(this.Rows, this.Cols);
            for (int i = 0; i < result.Rows; i++) {
                for (int j = 0; j < result.Cols; j++) {
                    this.Data[i, j] *= n.Data[i, j];
                }
            }
            return result;
        }

        /// <summary>
        /// Gets the scalar product
        /// </summary>
        /// <param name="n">Multiplier</param>
        /// <returns></returns>
        public Matrix Multiply(double n) {
            var result = new Matrix(this.Rows, this.Cols);
            for (int i = 0; i < result.Rows; i++) {
                for (int j = 0; j < result.Cols; j++) {
                    this.Data[i, j] *= n;
                }
            }
            return result;
        }

        /// <summary>
        /// Applies a function to every element of the matrix
        /// </summary>
        /// <param name="func"></param>
        public void Map(Func<double, double> func) {
            for (int i = 0; i < this.Rows; i++) {
                for (int j = 0; j < this.Cols; j++) {
                    this.Data[i, j] = func(this.Data[i, j]);
                }
            }
        }

        /// <summary>
        /// Applies a function to every element of the matrix
        /// </summary>
        /// <param name="func"></param>
        public static Matrix Map(Matrix m, Func<double, double> func) {
            var result = new Matrix(m.Rows, m.Cols);
            for (int i = 0; i < m.Rows; i++) {
                for (int j = 0; j < m.Cols; j++) {
                    var val = m.Data[i, j];
                    result.Data[i, j] = func(val);
                }
            }
            return result;
        }

        /// <summary>
        /// ToString override
        /// </summary>
        /// <returns></returns>
        public override string ToString() {
            return $"Matrix Dimenson: {this.Rows} by {this.Cols}";
        }

    }
}
