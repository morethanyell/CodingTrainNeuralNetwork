/*
 * A direct C# translation of Daniel Shiffman's Toy Neural Network Project
 * on his Git Repo link: http://bit.ly/2FjZBlD
 * which was originally written on JS
 * 
 * My name is Daniel L. Astillero
 * from the Philippines
 * 
 * */

using System.Collections.Generic;

namespace CodingTrainNeuralNetwork {

    /// <summary>
    /// The parent neural network class
    /// </summary>
    public class NeuralNetwork {

        /// <summary>
        /// The input data
        /// </summary>
        public readonly double[] InputNodes;

        /// <summary>
        /// The magic layer
        /// </summary>
        public readonly double[] HiddenNodes;

        /// <summary>
        /// The Network's prediction
        /// </summary>
        public readonly double[] OutputNodes;

        /// <summary>
        /// Weights input-hidden
        /// </summary>
        public Matrix WeightsIh;

        /// <summary>
        /// Weights hidden-output
        /// </summary>
        public Matrix WeightsHo;

        /// <summary>
        /// Bias to the hidden layers
        /// </summary>
        public Matrix BiasH;

        /// <summary>
        /// Bias to the output nodes
        /// </summary>
        public Matrix BiasO;

        /// <summary>
        /// Gets or sets the learning rate
        /// </summary>
        public double LearningRate { get; set; }

        /// <summary>
        /// Creates an instance of this Neural Network
        /// </summary>
        /// <param name="inputNodes">Defines the number of elements in the input nodes</param>
        /// <param name="hiddenNodes">Defines the number of hidden nodes</param>
        /// <param name="outputNodes">Defines the number of output nodes</param>
        public NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, double learningRate = 0.1) {
            this.InputNodes = new double[inputNodes];
            this.HiddenNodes = new double[hiddenNodes];
            this.OutputNodes = new double[outputNodes];

            this.WeightsIh = new Matrix(this.HiddenNodes.Length, this.InputNodes.Length);
            this.WeightsHo = new Matrix(this.OutputNodes.Length, this.HiddenNodes.Length);
            this.WeightsIh.Randomize();
            this.WeightsHo.Randomize();

            this.BiasH = new Matrix(this.HiddenNodes.Length, 1);
            this.BiasO = new Matrix(this.OutputNodes.Length, 1);
            this.BiasH.Randomize();
            this.BiasO.Randomize();

            this.LearningRate = learningRate;
        }

        /// <summary>
        /// Forward propagation
        /// </summary>
        public double[] FeedForward(double[] inputArray) {

            // Generating the Hidden Outputs
            var inputs = Matrix.FromArray(inputArray);
            var hidden = Matrix.Multiply(this.WeightsIh, inputs);
            hidden.Add(this.BiasH);
            // activation function!
            hidden.Map(NNUtilities.Sigmoid);

            // Generating the output's output!
            var output = Matrix.Multiply(this.WeightsHo, hidden);
            output.Add(this.BiasO);
            output.Map(NNUtilities.Sigmoid);

            // Sending back to the caller!
            return output.ToArray();
        }

        /// <summary>
        /// Back propagation
        /// </summary>
        /// <param name="inputArray"></param>
        /// <param name="targetArray"></param>
        public void Train(double[] inputArray, double[] targetArray) {

            // Generating the Hidden Outputs
            var inputs = Matrix.FromArray(inputArray);
            var hidden = Matrix.Multiply(this.WeightsIh, inputs);
            hidden.Add(this.BiasH);
            // activation function!
            hidden.Map(NNUtilities.Sigmoid);

            // Generating the output's output!
            var outputs = Matrix.Multiply(this.WeightsHo, hidden);
            outputs.Add(this.BiasO);
            outputs.Map(NNUtilities.Sigmoid);

            // Convert array to matrix object
            var targets = Matrix.FromArray(targetArray);

            // Calculate the error
            // ERROR = TARGETS - OUTPUTS
            var outputErrors = Matrix.Subtract(targets, outputs);

            // var gradient = outputs * (1 - outputs);
            // Calculate gradient
            var gradients = Matrix.Map(outputs, NNUtilities.DSigmoid);
            gradients.Multiply(outputErrors);
            gradients.Multiply(this.LearningRate);



            // Calculate deltas
            var hiddenT = Matrix.Transpose(hidden);
            var weightsHoDeltas = Matrix.Multiply(gradients, hiddenT);

            // Adjust the weights by deltas
            this.WeightsHo.Add(weightsHoDeltas);
            // Adjust the bias by its deltas (which is just the gradients)
            this.BiasO.Add(gradients);

            // Calculate the hidden gradient
            var wHoT = Matrix.Transpose(this.WeightsHo);
            var hiddenErrors = Matrix.Multiply(wHoT, outputErrors);

            // Calculate the hidden gradient
            var hiddenGradient = Matrix.Map(hidden, NNUtilities.DSigmoid);
            hiddenGradient.Multiply(hiddenErrors);
            hiddenGradient.Multiply(this.LearningRate);

            // Calculate the input->hidden deltas
            var inputsT = Matrix.Transpose(inputs);
            var weightIhDeltas = Matrix.Multiply(hiddenGradient, inputsT);

            this.WeightsIh.Add(weightIhDeltas);
            // Adjust teh bias by its deltas (which is just the gradients)
            this.BiasH.Add(hiddenGradient);

        }

    }


}
