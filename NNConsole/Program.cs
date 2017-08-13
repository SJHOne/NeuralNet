using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NN;

namespace NNConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            Network net = new Network();

            // Create the network topology Layer[0](2) x Layer[1](2) x Layer[2](2)
            net.CreateLayer(0, 2, NeuronType.Input);
            net.CreateLayer(1, 2, NeuronType.Hidden);
            net.CreateLayer(2, 2, NeuronType.Output);
            
            // Create the fully connected matrix of weights
            net.CompileNN();

            // Set the biases to match the worked example
            net.SetExplicitBias(0, 0, 0.0);
            net.SetExplicitBias(0, 1, 0.0);

            net.SetExplicitBias(1, 0, 0.35);
            net.SetExplicitBias(1, 1, 0.35);

            net.SetExplicitBias(2, 0, 0.60);
            net.SetExplicitBias(2, 1, 0.60);

            // Set the weights to match the worked example
            net.SetExplicitWeight(1, 0, 0, 0.15);
            net.SetExplicitWeight(1, 0, 1, 0.20);
            net.SetExplicitWeight(1, 1, 0, 0.25);
            net.SetExplicitWeight(1, 1, 1, 0.30);

            net.SetExplicitWeight(2, 0, 0, 0.40);
            net.SetExplicitWeight(2, 0, 1, 0.45);
            net.SetExplicitWeight(2, 1, 0, 0.50);
            net.SetExplicitWeight(2, 1, 1, 0.55);
            
            // Set the Inputs (top down)
            net.SetInput(0.05, 0.1);
            net.SetTargets(0.01, 0.99);
            net.Dump();

            Console.WriteLine("---------------------------------------------");

            // Run the network forward
            net.Run();
            Console.WriteLine("---------------------------------------------");
            net.Dump();
            Console.WriteLine("---------------------------------------------");

            // Run the network backwards (train) passing in the targets for the 2 output neurons (top down)
            net.Train();
            net.Dump();
            Console.WriteLine("---------------------------------------------");
            Console.WriteLine("");
            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();

            // Now run the network in a loop of 5000 trainings and see how low the error gets.
            for (int i = 0; i < 5000; i++)
            {
                Console.Write("[" + i.ToString() + "] ");
                net.Run();
                net.Train();
            }
            Console.WriteLine("Starting Total Error was : 0.2983711088");
            Console.WriteLine("---------------------------------------------");
            net.Dump();
            Console.WriteLine("---------------------------------------------");

            Console.WriteLine("");
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();

        }
    }
}
