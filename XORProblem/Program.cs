using CodingTrainNeuralNetwork;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace XORProblem {

    /// <summary>
    /// Console app for XOR
    /// </summary>
    public static class Program {

        /// <summary>
        /// Main
        /// </summary>
        /// <param name="args"></param>
        public static void Main(string[] args) {

            var nn = new NeuralNetwork(2, 16, 1);

            var guess1 = nn.FeedForward(new double[] { 0, 0 });
            var guess2 = nn.FeedForward(new double[] { 1, 0 });
            var guess3 = nn.FeedForward(new double[] { 0, 1 });
            var guess4 = nn.FeedForward(new double[] { 1, 1 });

            Console.WriteLine("\t\nPrior to training:\n");

            Console.WriteLine($"\tGuess for 0 XOR 0: {guess1[0]}");
            Console.WriteLine($"\tGuess for 1 XOR 0: {guess2[0]}");
            Console.WriteLine($"\tGuess for 0 XOR 1: {guess3[0]}");
            Console.WriteLine($"\tGuess for 1 XOR 1: {guess4[0]}");

            for (int i = 0; i < 2000; i++) {
                nn.Train(new double[] { 0, 0 }, new double[] { 0 });
                nn.Train(new double[] { 1, 0 }, new double[] { 1 });
                nn.Train(new double[] { 0, 1 }, new double[] { 1 });
                nn.Train(new double[] { 1, 1 }, new double[] { 0 });
            }

            guess1 = nn.FeedForward(new double[] { 0, 0 });
            guess2 = nn.FeedForward(new double[] { 1, 0 });
            guess3 = nn.FeedForward(new double[] { 0, 1 });
            guess4 = nn.FeedForward(new double[] { 1, 1 });

            Console.WriteLine("\nAfter training for 2,000 epochs:\n");

            Console.WriteLine($"\tGuess for 0 XOR 0: {guess1[0]}");
            Console.WriteLine($"\tGuess for 1 XOR 0: {guess2[0]}");
            Console.WriteLine($"\tGuess for 0 XOR 1: {guess3[0]}");
            Console.WriteLine($"\tGuess for 1 XOR 1: {guess4[0]}");

            Console.ReadKey();

        }
    }
}
