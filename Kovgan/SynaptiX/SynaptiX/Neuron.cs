using System;

namespace NeuroGPT_Kovgan_PI31_2
{
    // Класс одного нейрона
    class Neuron
    {
        // Поля
        private NeuronType type;     // тип нейрона
        private double[] weights;    // веса: weights[0] — порог, weights[j + 1] — вес j-го входа
        private double[] inputs;     // входы
        private double output;       // выход
        private double derivative;   // производная функции активации

        // Свойства
        public double[] Weights { get => weights; set => weights = value; }
        public double[] Inputs { get => inputs; set => inputs = value; }
        public double Output { get => output; }
        public double Derivative { get => derivative; }

        // Конструктор
        public Neuron(double[] memoryWeights, NeuronType typeNeuron)
        {
            type = typeNeuron;
            weights = memoryWeights;
        }

        // Активация нейрона: аффинное преобразование и нелинейность
        public void Activator(double[] i)
        {
            inputs = i; // вектор входного сигнала

            double sum = weights[0]; // порог

            for (int j = 0; j < inputs.Length; j++)
            {
                sum += inputs[j] * weights[j + 1];
            }

            switch (type)
            {
                case NeuronType.Hidden: // нейрон скрытого слоя: логистическая функция (сигмоида)
                    output = Logistich(sum);
                    derivative = Logistich_Derivativator(sum);
                    break;

                case NeuronType.Output: // нейрон выходного слоя: экспонента, нормировка (softmax) в OutputLayer
                    output = Math.Exp(sum);
                    break;
            }
        }

        // Логистическая функция активации
        public double Logistich(double sum)
        {
            return 1.0 / (1.0 + Math.Exp(-sum));
        }

        // Производная логистической функции
        public double Logistich_Derivativator(double sum)
        {
            double s = Logistich(sum);
            return s * (1.0 - s);
        }
    }
}