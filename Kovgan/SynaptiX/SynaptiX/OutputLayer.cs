namespace NeuroGPT_Kovgan_PI31_2
{
    // Выходной слой.
    // В нашем варианте содержит 10 нейронов, активация — softmax.
    class OutputLayer : Layer
    {
        public OutputLayer(int non, int nopn, NeuronType nt, string nm_Layer) :
            base(non, nopn, nt, nm_Layer)
        { }

        // Прямой проход: нормировка экспонент (softmax) в фактический выход сети
        public override void Recognize(Network net, Layer nextlayer)
        {
            double e_sum = 0;
            for (int i = 0; i < neurons.Length; i++)
                e_sum += neurons[i].Output;

            for (int i = 0; i < neurons.Length; i++)
                net.Fact[i] = neurons[i].Output / e_sum;
        }

        // Обратный проход.
        // errors[i] = желаемый отклик - фактический (для softmax это и есть локальный градиент).
        // Возвращает градиент по выходам предыдущего слоя.
        public override double[] BackwardPass(double[] errors)
        {
            // Градиент для предыдущего слоя считаем до коррекции весов
            double[] gr_sum = InputGradient(errors);

            CorrectWeights(errors);
            return gr_sum;
        }
    }
}
