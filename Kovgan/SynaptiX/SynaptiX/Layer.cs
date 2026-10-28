using System;
using System.Globalization;
using System.IO;

namespace NeuroGPT_Kovgan_PI31_2
{
    // Базовый класс слоя нейронной сети
    abstract class Layer
    {
        // Поля
        protected string name_Layer;
        string pathDirWeights;                 // каталог файлов синаптических весов
        string pathFileWeights;                // файл синаптических весов этого слоя
        protected int numofneurons;            // число нейронов текущего слоя
        protected int numofprevneurons;        // число нейронов предыдущего слоя
        protected const double learningrate = 0.060; // скорость обучения
        protected const double momentum = 0.050d;    // момент инерции
        protected double[,] lastdeltaweights;  // поправки весов предыдущей итерации
        protected Neuron[] neurons;            // нейроны текущего слоя

        // Свойства
        public Neuron[] Neurons { get => neurons; set => neurons = value; }
        public string MemoryPath { get => pathFileWeights; }
        public double[] Data // передача входных данных на нейроны слоя и их активация
        {
            set
            {
                for (int i = 0; i < numofneurons; i++)
                {
                    Neurons[i].Activator(value);
                }
            }
        }

        // Конструктор
        protected Layer(int non, int nopn, NeuronType nt, string nm_Layer)
        {
            numofneurons = non;
            numofprevneurons = nopn;
            Neurons = new Neuron[non];
            name_Layer = nm_Layer;
            pathDirWeights = AppDomain.CurrentDomain.BaseDirectory + "memory\\";
            pathFileWeights = pathDirWeights + name_Layer + "_memory.csv";

            lastdeltaweights = new double[non, nopn + 1];
            double[,] Weights;

            if (File.Exists(pathFileWeights))
                Weights = WeightInitialize(MemoryMode.GET, pathFileWeights);
            else
            {
                Directory.CreateDirectory(pathDirWeights);
                Weights = WeightInitialize(MemoryMode.INIT, pathFileWeights);
            }

            for (int i = 0; i < non; i++)
            {
                double[] tmp_weights = new double[nopn + 1];
                for (int j = 0; j < nopn + 1; j++)
                {
                    tmp_weights[j] = Weights[i, j];
                }
                Neurons[i] = new Neuron(tmp_weights, nt);
            }
        }

        // Работа с массивом синаптических весов слоя
        public double[,] WeightInitialize(MemoryMode mm, string path)
        {
            char[] delim = new char[] { ';' };
            double[,] weights = new double[numofneurons, numofprevneurons + 1];

            switch (mm)
            {
                case MemoryMode.GET:
                    string[] lines = File.ReadAllLines(path);

                    // Файл от другой архитектуры сети не подходит: создаём веса заново
                    if (!IsShapeValid(lines, delim))
                        return WeightInitialize(MemoryMode.INIT, path);

                    for (int i = 0; i < numofneurons; i++)
                    {
                        string[] memory_element = lines[i].Split(delim);
                        for (int j = 0; j < numofprevneurons + 1; j++)
                        {
                            weights[i, j] = double.Parse(memory_element[j], CultureInfo.InvariantCulture);
                        }
                    }
                    break;

                case MemoryMode.SET:
                    // Запись текущих весов нейронов в файл (перезапись)
                    Directory.CreateDirectory(Path.GetDirectoryName(path));
                    using (StreamWriter sw = new StreamWriter(path, false))
                    {
                        for (int i = 0; i < numofneurons; i++)
                        {
                            for (int j = 0; j < numofprevneurons + 1; j++)
                            {
                                sw.Write(neurons[i].Weights[j].ToString("R", CultureInfo.InvariantCulture));

                                if (j != numofprevneurons) sw.Write(";");
                            }
                            sw.WriteLine();
                        }
                    }
                    break;

                case MemoryMode.INIT:
                    // Равномерные случайные веса из [-1/sqrt(n), 1/sqrt(n)], n — число входов
                    Random random = new Random();
                    double limit = 1.0 / Math.Sqrt(numofprevneurons);

                    for (int i = 0; i < numofneurons; i++)
                        for (int j = 0; j < numofprevneurons + 1; j++)
                            weights[i, j] = (random.NextDouble() * 2.0 - 1.0) * limit;

                    // Сохраняем начальные веса в файл памяти
                    Directory.CreateDirectory(Path.GetDirectoryName(path));
                    using (StreamWriter sw = new StreamWriter(path, false))
                    {
                        for (int i = 0; i < numofneurons; i++)
                        {
                            for (int j = 0; j < numofprevneurons + 1; j++)
                            {
                                sw.Write(weights[i, j].ToString("R", CultureInfo.InvariantCulture));

                                if (j != numofprevneurons) sw.Write(";");
                            }
                            sw.WriteLine();
                        }
                    }
                    break;
            }
            return weights;
        }

        // Проверка, что размер таблицы в файле совпадает с размером слоя
        private bool IsShapeValid(string[] lines, char[] delim)
        {
            if (lines.Length < numofneurons) return false;

            for (int i = 0; i < numofneurons; i++)
                if (lines[i].Split(delim).Length != numofprevneurons + 1) return false;

            return true;
        }

        // Коррекция весов слоя по локальным градиентам delta[i] с моментом инерции
        protected void CorrectWeights(double[] delta)
        {
            for (int i = 0; i < numofneurons; i++)
            {
                for (int n = 0; n < numofprevneurons + 1; n++)
                {
                    double deltaw;
                    if (n == 0) // порог
                        deltaw = momentum * lastdeltaweights[i, 0] + learningrate * delta[i];
                    else
                        deltaw = momentum * lastdeltaweights[i, n] + learningrate * neurons[i].Inputs[n - 1] * delta[i];

                    lastdeltaweights[i, n] = deltaw;
                    neurons[i].Weights[n] += deltaw;
                }
            }
        }

        // Градиент по входам слоя (то есть по выходам предыдущего слоя), по текущим весам
        protected double[] InputGradient(double[] delta)
        {
            double[] gr_sum = new double[numofprevneurons];

            for (int j = 0; j < numofprevneurons; j++)
            {
                double sum = 0;
                for (int k = 0; k < numofneurons; k++)
                    sum += neurons[k].Weights[j + 1] * delta[k];

                gr_sum[j] = sum;
            }
            return gr_sum;
        }

        abstract public void Recognize(Network net, Layer nextlayer);
        abstract public double[] BackwardPass(double[] stuff);
    }
}
