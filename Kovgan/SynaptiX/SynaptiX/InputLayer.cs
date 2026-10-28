using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace NeuroGPT_Kovgan_PI31_2
{
    // Входной слой: обучающая и тестовая выборки.
    // Обучающая строка: метка цифры и 15 пикселей (16 чисел через пробел).
    // Тестовая строка: 15 пикселей (как её пишет кнопка тестового примера).
    class InputLayer
    {
        public double[,] trainset; // обучающая выборка
        public double[,] testset;  // тестовая выборка

        // Свойства
        public double[,] Trainset { get => trainset; }
        public double[,] Testset { get => testset; }

        // Конструктор
        public InputLayer(NetworkMode nm)
        {
            string path = AppDomain.CurrentDomain.BaseDirectory + "Data\\";

            switch (nm)
            {
                case NetworkMode.Train:
                    trainset = ReadSet(path + "train.txt");
                    Shuffling_Array_Rows(trainset); // перетасовка методом Фишера-Йетса
                    break;

                case NetworkMode.Test:
                    testset = ReadSet(path + "test.txt");
                    Shuffling_Array_Rows(testset);
                    break;
            }
        }

        // Чтение выборки из файла, пустые строки пропускаются
        private double[,] ReadSet(string file)
        {
            if (!File.Exists(file))
                throw new FileNotFoundException("Файл выборки не найден: " + file);

            List<double[]> rows = new List<double[]>();
            foreach (string line in File.ReadAllLines(file))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;

                string[] tmpStr = line.Trim().Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                double[] row = new double[tmpStr.Length];
                for (int j = 0; j < tmpStr.Length; j++)
                    row[j] = double.Parse(tmpStr[j], CultureInfo.InvariantCulture);
                rows.Add(row);
            }

            if (rows.Count == 0)
                throw new InvalidOperationException("Выборка пуста: " + file);

            double[,] set = new double[rows.Count, rows[0].Length];
            for (int i = 0; i < rows.Count; i++)
                for (int j = 0; j < rows[0].Length; j++)
                    set[i, j] = rows[i][j];
            return set;
        }

        // Перетасовка строк массива методом Фишера-Йетса (Fisher-Yates)
        public void Shuffling_Array_Rows(double[,] arr)
        {
            Random rnd = new Random();
            int rows = arr.GetLength(0);
            int cols = arr.GetLength(1);

            for (int i = rows - 1; i > 0; i--)
            {
                int j = rnd.Next(i + 1); // случайный индекс от 0 до i
                for (int k = 0; k < cols; k++)
                {
                    double tmp = arr[i, k];
                    arr[i, k] = arr[j, k];
                    arr[j, k] = tmp;
                }
            }
        }
    }
}
