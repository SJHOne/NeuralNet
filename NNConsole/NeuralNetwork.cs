///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// After reasearching neural networks for a while, I thought I should test my knowledge by writing an implementation
// from scratch. This is my first attempt.
//
// Inspiration for this neural network implementation came from the following source:
// [1] https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
// This network has been tested against the worked example given above
//
// Additional sources:
// [2] https://page.mi.fu-berlin.de/rojas/neural/chapter/K7.pdf
// [3] https://jamesmccaffrey.wordpress.com/2016/12/14/deriving-the-gradient-for-neural-network-back-propagation-with-cross-entropy-error/
// [4] https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1
//
//  - Steve Hobley 2017
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
using System;
using System.Collections.Generic;
using System.Linq;

namespace NN
{
    /// <summary>
    /// A Neural Network is made up of an ordered list of Layers
    /// </summary>
    public class Network
    {
        private SortedList<int, Layer> nn;

        public Network()
        {
            nn = new SortedList<int, Layer>();
        }

        /// <summary>
        /// Call this to create an additional layer in the network
        /// </summary>
        /// <param name="rank">0 based index, the number of the layer</param>
        /// <param name="noofNeurons">The number of neurons this layer should contain</param>
        /// <param name="type">Input/Hidden/Output, there should be only 1 Input and 1 Output layer.</param>
        public void CreateLayer(int rank, int noofNeurons, NeuronType type)
        {
            nn.Add(rank, new Layer(noofNeurons, type));
        }

        /// <summary>
        /// Call this to set the explicit weight of an INPUT connection in the network
        /// </summary>
        /// <param name="layer">0 based index</param>
        /// <param name="neuron">0 based index</param>
        /// <param name="connection">o based</param>
        /// <param name="Value">The new value for the weight</param>
        public void SetExplicitWeight(int layer, int neuron, int connection, double Value)
        {
            nn[layer].neurons[neuron].inputconnections[connection].weight = Value;
        }

        /// <summary>
        /// Call this function to set an explicit bias value to any neuron.
        /// </summary>
        /// <param name="layer">0 based index</param>
        /// <param name="neuron">0 based index</param>
        /// <param name="Value">The new value for the bias</param>
        public void SetExplicitBias(int layer, int neuron, double Value)
        {
            nn[layer].neurons[neuron].SetBiasExplicit(Value);
        }

        /// <summary>
        /// Optional method to set the target values on the OUTPUT neurons.
        /// These values can be passed in via a call to Train(...)
        /// </summary>
        /// <param name="list">An array of doubles, one for each output neuron.</param>
        public void SetTargets(params double[] list)
        {
            for (int i = 0; i < list.Length; i++)
                nn[nn.Count - 1].neurons[i].Target = list[i];
        }

        /// <summary>
        /// Builds all the connections between the input, hidden and output layers. 
        /// CreateLayer() should not be called after this.
        /// </summary>
        public void CompileNN()
        {
            int layercount = nn.Count();

            if (layercount < 1)
                return;

            // Create all the connections in each layer
            // Is there a previous layer? wire up connections to this layer, adjusting for number of neurons
            // Is there a next layer? Do the same, making same adjustment
            for (int i = 1; i < layercount; i++)
                nn[i].ConnectLayer(nn[i-1]);
        }

        /// <summary>
        /// Sets the inputs to the network.
        /// </summary>
        /// <param name="list">An set of doubles, one for each input neuron</param>
        public void SetInput(params double[] list)
        {
            if (nn.Count == 0)
                throw new Exception("This network has no layers!");

            nn[0].SetValues(list);
        }

        /// <summary>
        /// Returns the output values, in an array
        /// </summary>
        /// <returns></returns>
        public double[] ReadOutput()
        {
            if (nn.Count == 0)
                return null;

            var outputList = nn[nn.Count - 1].neurons;
            return outputList.Select(x => x.OutputValue).ToArray<double>();
        }

        /// <summary>
        /// Performs a forward run on the network, topology should be defined, as well as all connection
        /// and input values set.
        /// </summary>
        public void Run()
        {
            int layercount = nn.Count();
            
            // Run through the neurons in rank order and Compute() them
            for (int i = 0; i < layercount; i++)
                nn[i].ComputeLayer();
            
            // Finally, show the error
            double totalError = nn[nn.Count - 1].ComputeTotalError();
            Console.WriteLine("Total Error: " + totalError.ToString("0.0000000000"));

        }

        /// <summary>
        /// The tricky part - pass in a list of expected outputs and this function will run the network
        /// backwards to adjust all the weights by the 'loss gradient'
        /// </summary>
        /// <param name="expectedvalues">A variable number of double precision numbers, one for each output neuron</param>
        public void Train()
        {
            // Starting at the output layer, create a new candidate value for connection weights.
            // We have to calculate the new weights using existing values, then commit them all en masse later.
            //nn[nn.Count - 1].SetTargets(expectedvalues);

            // Go back one layer, adjust all connections
            for (int i = nn.Count - 1; i > 0; i--) // [0] is the input layer, so skip that
            {
                nn[i].TrainLayer();
            }

            // Commit all connections with the new weights.
            for (int i = nn.Count - 1; i > 0; i--) // [0] is the input layer, so skip that
            {
                nn[i].CommitLayer();
            }
        }

        /// <summary>
        /// Displays detailed information on the state of all layers, neurons, and connections.
        /// </summary>
        public void Dump()
        {
            int i = nn.Count;
            Console.WriteLine("Network has [" + i.ToString() + "] layers.");
            for (int j = 0; j < i; j++)
            {
                Console.WriteLine("\tLayer [" + j.ToString() + "]:");
                nn[j].Dump();
            }
        }
    }
    
    /// <summary>
    /// A Layer is made up of an ordered list of neurons 
    /// </summary>
    public class Layer
    {
        public List<Neuron> neurons;
        public NeuronType LayerType;

        public Layer(int noofNeurons, NeuronType layertype)
        {
            // Use Identity for the input layer, sigmoid for the hidden and output layers.
            ActivationFunctions functionType = ActivationFunctions.Identity;

            switch (layertype)
            {
                case NeuronType.Hidden:
                case NeuronType.Output:
                    functionType = ActivationFunctions.Sigmoid;
                    break;
            }

            neurons = new List<Neuron>();
            for (int i = 0; i < noofNeurons; i++)
                neurons.Add(new Neuron(functionType));

            LayerType = layertype;
        }

        public void ComputeLayer()
        {
            foreach (Neuron nn in neurons)
            {
                nn.Compute();
            }
        }

        /// <summary>
        /// Connects every neuron in this layer to the previous layer
        /// </summary>
        /// <param name="previous">The previous ranked layer</param>
        public void ConnectLayer(Layer previous)
        {
            if (neurons.Count == 0)
                return;

            if (previous.neurons.Count == 0)
                return;

            foreach (Neuron dest in neurons)
            {
                foreach (Neuron source in previous.neurons)
                {
                    Connection conn = new Connection();
                    conn.source = source;
                    conn.destination = dest;
                    conn.weight = Randomizer.GetRandomWeight(0.0, 1.0);

                    source.outputconnections.Add(conn);
                    dest.inputconnections.Add(conn);
                }
            }
        }

        public double ComputeTotalError()
        {
            double totalError = 0.0;

            foreach (Neuron n in neurons)
                totalError += n.ComputeError();

            return totalError;
        }

        public void SetTargets(params double[] list)
        {
            if (list.Length != neurons.Count)
                throw new Exception("List of targets length does not match neuron count");
            int i = 0;

            foreach (Neuron nn in neurons)
                nn.Target = list[i++];
        }

        public void TrainLayer()
        {
            foreach (Neuron nn in neurons)
                foreach (Connection nc in nn.inputconnections)
                    nc.CalculateUpdateWeight();
        }

        public void CommitLayer()
        {
            // Commit every input connection 
            foreach (Neuron nn in neurons)
                foreach (Connection nc in nn.inputconnections)
                    nc.CommitNewWeight();
        }

        public void SetValues(double[] list)
        {
            if (list.Length != neurons.Count)
                throw new Exception("Inputs passed do not match  of in of input neurons!");

            int i = 0;
            foreach (Neuron n in neurons)
                n.SetNetInputValueExplicit(list[i++]);
        }

        public void Dump()
        {
            int k = 0;
            Console.WriteLine("\tNeuron count = " + this.neurons.Count.ToString() + ":");
            foreach (Neuron nn in neurons)
            {
                Console.WriteLine("\t\tNeuron[" + k.ToString() + "]: ");
                nn.Dump();
                k++;
            }
        }
    }

    /// <summary>
    /// A Neuron has storage for net input, bias, output, activiation function type, 
    /// inputconnection (list), and output connection (list)
    /// </summary>
    public class Neuron
    {
        private ActivationFunctions activationFunction = ActivationFunctions.Identity;
        private double outputValue = 0.0;
        private double netInputValue = 0.0;
        private double internalBias = 0.0;   // Should this be random at the start? [Yes]

        // Backpropagation properties
        double howMuchOutputChangeWithRespectToInput = 0.0;
        public double Target = 0.0;

        public List<Connection> inputconnections;
        public List<Connection> outputconnections;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="type">Defines the activation function to use</param>
        public Neuron(ActivationFunctions type)
        {
            //Define the ActivationFunction
            activationFunction = type;
            inputconnections = new List<Connection>();
            outputconnections = new List<Connection>();
            internalBias = Randomizer.GetRandomWeight(0.0, 1.0);   // Set this to random at the start

        }

        #region Backpropagation Functions

        public double DeltaOutputWithRespectToInput()
        {
            if (howMuchOutputChangeWithRespectToInput != 0.0)
                return howMuchOutputChangeWithRespectToInput;

            howMuchOutputChangeWithRespectToInput = ActivationFunctionImpl.SigmoidDerivative(OutputValue);
            return howMuchOutputChangeWithRespectToInput;
        }
        public double DeltaTotalError()
        {
            return OutputValue - Target;
        }
        public void SetNetInputValueExplicit(double Value)
        {
            outputValue = Value;
        }
        public double ComputeError()
        {
            double diff = (Target - outputValue);
            return 0.5 * diff * diff;
        }

        #endregion

        public void SetBiasExplicit(double Value)
        {
            internalBias = Value;
        }

        public void Compute()
        {
            // Sum the input connections and set internal value via the activation function
            // Each layer should be fully connected to each other layer.

            // If this is just an input, then do nothing, preserve the outputValue as is
            if (inputconnections.Count == 0)
                return;

            netInputValue = 0.0;

            // Sum each input connection
            foreach (Connection nc in inputconnections)
                netInputValue += (nc.source.outputValue * nc.weight);

            // Do you just add the internal bias?
            netInputValue += internalBias;

            // Then pass the value to the correct activation function and update the result
            outputValue = ActivationFunctionImpl.Run(activationFunction, netInputValue);
        }

        public double OutputValue
        {
            get { return outputValue; }
            private set { }
        }

        public void Dump()
        {
            Console.Write("\t\t\tNumber of inputs : " + this.inputconnections.Count.ToString() + " [");
            
            // Dump connection weights
            foreach (Connection nnc in inputconnections)
                Console.Write(nnc.weight.ToString("0.00000000") + " ");
            Console.WriteLine("]");
            Console.WriteLine("\t\t\tNumber of outputs : " + this.outputconnections.Count.ToString());
            
            // Dump net input, output bias, and function
            Console.WriteLine("\t\t\tActivation function : " + this.activationFunction.ToString());
            Console.WriteLine("\t\t\tInternal Bias Value : " + this.internalBias.ToString("0.00000000"));
            Console.WriteLine("\t\t\tNet Input Value : " + this.netInputValue.ToString("0.00000000"));
            Console.WriteLine("\t\t\tOutput Value : " + this.outputValue.ToString("0.00000000"));

            if (outputconnections.Count == 0)
                Console.WriteLine("\t\t\tTarget Value : " + this.Target.ToString("0.00000000"));
        }
    }
    
    /// <summary>
    /// A connection has a refence to source Neuron, 
    /// destination Neuron, weight 
    /// and new Weight - for deferred Commit()
    /// </summary>
    public class Connection
    {
        public Neuron source = null;
        public Neuron destination = null;
        public double weight = 0.0;
        private double newWeight = 0.0;

        public double gradient = 0.0;       // Used for backpropagation, and calculating the error
        public double learningRate = 0.5;   // Constant Learning Rate, for now.

        public void CalculateUpdateWeight()
        {
            // 'We perform the actual updates in the neural network after we have the new weights 
            // leading into the hidden layer neurons (ie, we use the original weights, not the 
            // updated weights, when we continue the backpropagation algorithm below).' - from reference [1] above.

            // Figure out if we are a connection to the output layer.
            if (this.destination.outputconnections.Count == 0)
            {
                // This connection leads to an output neuron:

                // How much does total (output) error change with respect to the output
                //destination.Target = target;
                double howMuchTotalErrorChange = destination.DeltaTotalError();

                // Use the derivative of the sigmoid function to calculate the next step
                double howMuchOutputChangeWithRespectToInput = destination.DeltaOutputWithRespectToInput();

                // How much does the total net *input* of destination change due to this connection
                double howMuchTotalNetInputChangeDueToThisWeight = this.source.OutputValue;

                double totalWeightChangeForThisConnection = howMuchTotalErrorChange * howMuchOutputChangeWithRespectToInput * howMuchTotalNetInputChangeDueToThisWeight;
                
                // This will be 'locked in' with a commit, later
                newWeight = weight - (totalWeightChangeForThisConnection * learningRate);
            }
            else
            {
                // Must be a connection leading to a hidden neuron, (this is a bit trickier)

                // 'We’re going to use a similar process as we did for the output layer, 
                // but slightly different to account for the fact that the output of each 
                // hidden layer neuron contributes to the output (and therefore error) 
                // of multiple output neurons. We know that out_{h1} affects both out_{o1} and out_{o2} 
                // therefore the \frac{\partial E_{total}}{\partial out_{h1}} needs to 
                // take into consideration its effect on the both output neurons:' - from reference [1] above.

                // How many output neurons are affected by this connection?
                int noofNeurons = this.destination.outputconnections.Count;

                double runningTotal = 0.0;

                // iterate over these neurons and gather the 'effect' on each.
                foreach (Connection nnc in this.destination.outputconnections)
                {
                    Neuron nOutputNeuron = nnc.destination;
                    runningTotal += (nOutputNeuron.DeltaTotalError() * nOutputNeuron.DeltaOutputWithRespectToInput() * nnc.weight);
                }

                double partialDerivativeDestination = ActivationFunctionImpl.SigmoidDerivative(this.destination.OutputValue);
                double partialDerivativeNetInput = source.OutputValue;
                
                // This will be 'locked in' with a commit, later
                newWeight = weight - (runningTotal * partialDerivativeDestination * partialDerivativeNetInput * learningRate);
            }
        }

        public void CommitNewWeight()
        {
            weight = newWeight;
        }
    }

    #region Enums

    public enum NeuronType
    {
        Input,
        Hidden,
        Output
    }

    // https://stats.stackexchange.com/questions/115258/comprehensive-list-of-activation-functions-in-neural-networks-with-pros-cons
    public enum ActivationFunctions
    {
        Identity,
        Step,
        PiecewiseLinear,
        Sigmoid,
        Bipolar,
        LogLog,
        BipolarSigmoid,
        Tanh,
        Absolute,
        Rectifier
    }

    #endregion

    #region Static Functions

    public static class ActivationFunctionImpl
    {
        public static double Run(ActivationFunctions function, double value)
        {
            switch (function)
            {
                case ActivationFunctions.Identity:
                    return Identity(value);

                case ActivationFunctions.Bipolar:
                    return Bipolar(value);

                case ActivationFunctions.Absolute:
                    return Absolute(value);

                case ActivationFunctions.BipolarSigmoid:
                    return BipolarSigmoid(value);

                case ActivationFunctions.LogLog:
                    return LogLog(value);

                case ActivationFunctions.PiecewiseLinear:
                    return PiecewiseLinear(value);

                case ActivationFunctions.Rectifier:
                    return Rectifier(value);

                case ActivationFunctions.Sigmoid:
                    return Sigmoid(value);

                case ActivationFunctions.Step:
                    return Step(value);

                case ActivationFunctions.Tanh:
                    return Tanh(value);
            }

            return 0.0;
        }

        public static double Identity(double input)
        {
            return input;
        }

        public static double Step(double input)
        {
            if (input <= 0.0)
                return 0.0;
            else
                return 1.0;
        }

        public static double PiecewiseLinear(double input)
        {
            throw new Exception("PiecewiseLinear() not implemented.");
        }

        /////////////////////////////////////////////////////////////////////////////
        // http://www.robosoup.com/2008/09/sigmoid-function-in-c.html
        public static double Sigmoid(double input)
        {
            return 1 / (1 + Math.Exp(-input));
        }

        /// Used to calculate the error gradient, when using the sigmoid function.
        public static double SigmoidDerivative(double x)
        {
            return x * (1 - x);
        }
        /////////////////////////////////////////////////////////////////////////////

        public static double Bipolar(double input)
        {
            throw new Exception("Bipolar() not implemented.");
        }

        public static double LogLog(double input)
        {
            throw new Exception("LogLog() not implemented.");
        }

        public static double BipolarSigmoid(double input)
        {
            throw new Exception("BipolarSigmoid() not implemented.");
        }

        public static double Tanh(double input)
        {
            return Math.Tanh(input);
        }

        public static double Absolute(double input)
        {
            if (input < 0.0)
                input = input * -1;

            return input;
        }

        public static double Rectifier(double input)
        {
            if (input <= 0.0)
                return 0.0;
            else
                return input;
        }
        
        /// <summary>
        /// Used to calculate the error gradient, when using the sigmoid function.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public static double PythonSigmoidDerivative(double input)
        {
            return input * (1 - input);
        }


    }

    public static class Randomizer
    {
        // Only one instance of Random needed.
        static Random random = new Random();

        public static double GetRandomWeight(double minimum, double maximum)
        {
            return random.NextDouble() * (maximum - minimum) + minimum;
        }
    }
    
    #endregion
}
