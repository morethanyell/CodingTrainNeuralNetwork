using System;

namespace CodingTrainNeuralNetwork {

    /// <summary>
    /// Helper methods
    /// </summary>
    public static class NNUtilities {

        /// <summary>
        /// A sigmoid function is a mathematical function having a characteristic "S"-shaped curve or sigmoid curve.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static double Sigmoid(double x) {
            return 1 / (1 + Math.Exp(-x));
        }

        /// <summary>
        /// The derivative of Sigmoid
        /// </summary>
        /// <param name="y"></param>
        /// <returns></returns>
        public static double DSigmoid(double y) {
            return y * (1 - y);
        }

    }
}
