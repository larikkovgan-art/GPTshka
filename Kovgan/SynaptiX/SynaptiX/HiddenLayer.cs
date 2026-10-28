namespace NeuroGPT_Kovgan_PI31_2
{
    // Скрытый слой нейронной сети.
    class HiddenLayer : Layer
    {
        public HiddenLayer(int non, int nopn, NeuronType nt, string nm_Layer) :
            base(non, nopn, nt, nm_Layer)
        { }

        // Прямой проход: выходы слоя становятся входами следующего слоя
        public override void Recognize(Network net, Layer nextlayer)
        {
            double[] hidden_out = new double[numofneurons];
            for (int i = 0; i < numofneurons; i++)
                hidden_out[i] = neurons[i].Output;
            nextlayer.Data = hidden_out;
        }

        // Обратный проход.
        // gr_sums[i] — градиент ошибки по выходу i-го нейрона этого слоя.
        // Возвращает градиент по выходам предыдущего слоя.
        public override double[] BackwardPass(double[] gr_sums)
        {
            // Локальный градиент нейрона: градиент по выходу * производная активации
            double[] delta = new double[numofneurons];
            for (int i = 0; i < numofneurons; i++)
                delta[i] = gr_sums[i] * neurons[i].Derivative;

            // Градиент для предыдущего слоя считаем до коррекции весов
            double[] gr_sum = InputGradient(delta);

            CorrectWeights(delta);
            return gr_sum;
        }
    }
}
