namespace NeuroGPT_Kovgan_PI31_2
{
    // Нейронная сеть прямого распространения.
    //
    // Архитектура нашего варианта:
    // 15 -> 71 -> 30 -> 10
    class Network
    {
        private InputLayer input_layer = null;
        private HiddenLayer hidden_layer1 = new HiddenLayer(71, 15, NeuronType.Hidden, nameof(hidden_layer1));
        private HiddenLayer hidden_layer2 = new HiddenLayer(30, 71, NeuronType.Hidden, nameof(hidden_layer2));
        private OutputLayer output_layer = new OutputLayer(10, 30, NeuronType.Output, nameof(output_layer));

        private double[] fact = new double[10]; // фактический выход сети
        public double[] e_error_avr;            // средняя энергия ошибки по эпохам

        // Свойства
        public double[] Fact { get => fact; }  // массив фактического выхода сети
        // Среднее значение энергии ошибки эпохи обучения
        public double[] E_error_avr { get => e_error_avr; set => e_error_avr = value; }

        // Конструктор
        public Network() { }

        // Прямой проход
        public void ForwardPass(Network net, double[] netInput)
        {
            net.hidden_layer1.Data = netInput;
            net.hidden_layer1.Recognize(null, net.hidden_layer2);
            net.hidden_layer2.Recognize(null, net.output_layer);
            net.output_layer.Recognize(net, null);
        }

        // Обучение методом обратного распространения ошибки
        public void Train(Network net)
        {
            net.input_layer = new InputLayer(NetworkMode.Train); // обучающая выборка
            int epoches = 10;       // количество эпох обучения
            double tmpSumError;     // сумма энергий ошибки по выходам для одного образа
            double[] errors;        // сигнал ошибки выходного слоя
            double[] temp_gsums1;   // градиент по выходам 1-го скрытого слоя
            double[] temp_gsums2;   // градиент по выходам 2-го скрытого слоя

            e_error_avr = new double[epoches];

            for (int k = 0; k < epoches; k++) // перебор эпох обучения
            {
                e_error_avr[k] = 0; // в начале эпохи средняя энергия ошибки обнуляется
                net.input_layer.Shuffling_Array_Rows(net.input_layer.Trainset); // перетасовка выборки

                for (int i = 0; i < net.input_layer.Trainset.GetLength(0); i++)
                {
                    // Образ: столбец 0 — метка, столбцы 1..15 — пиксели
                    double[] tmpTrain = new double[15];
                    for (int j = 0; j < tmpTrain.Length; j++)
                        tmpTrain[j] = net.input_layer.Trainset[i, j + 1];

                    // Прямой проход
                    ForwardPass(net, tmpTrain);

                    // Ошибка по образу
                    tmpSumError = 0;
                    errors = new double[net.fact.Length];
                    for (int x = 0; x < errors.Length; x++)
                    {
                        if (x == net.input_layer.Trainset[i, 0]) // номер нейрона совпадает с желаемым откликом
                            errors[x] = 1.0 - net.fact[x];
                        else
                            errors[x] = 0.0 - net.fact[x];

                        tmpSumError += errors[x] * errors[x] / 2;
                    }
                    e_error_avr[k] += tmpSumError / errors.Length;

                    // Обратный проход и коррекция весов
                    temp_gsums2 = net.output_layer.BackwardPass(errors);
                    temp_gsums1 = net.hidden_layer2.BackwardPass(temp_gsums2);
                    net.hidden_layer1.BackwardPass(temp_gsums1);
                }
                e_error_avr[k] /= net.input_layer.Trainset.GetLength(0); // средняя энергия ошибки эпохи
            }
            net.input_layer = null; // уборка входного слоя

            // Запись скорректированных весов в память
            net.hidden_layer1.WeightInitialize(MemoryMode.SET, net.hidden_layer1.MemoryPath);
            net.hidden_layer2.WeightInitialize(MemoryMode.SET, net.hidden_layer2.MemoryPath);
            net.output_layer.WeightInitialize(MemoryMode.SET, net.output_layer.MemoryPath);
        }
    }
}