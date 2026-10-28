using System;
using System.Drawing;
using System.IO;
using System.Windows.Forms;

namespace NeuroGPT_Kovgan_PI31_2
{
    public partial class Form1 : Form
    {
        // Массив из 15 входных пикселей
        private double[] inputPixels = new double[15];

        public Form1()
        {
            InitializeComponent();

            // Подключаем один обработчик
            // ко всем 15 кнопкам-пикселям
            button1.Click += PixelButton_Click;
            button2.Click += PixelButton_Click;
            button3.Click += PixelButton_Click;
            button4.Click += PixelButton_Click;
            button5.Click += PixelButton_Click;
            button6.Click += PixelButton_Click;
            button7.Click += PixelButton_Click;
            button8.Click += PixelButton_Click;
            button9.Click += PixelButton_Click;
            button10.Click += PixelButton_Click;
            button11.Click += PixelButton_Click;
            button12.Click += PixelButton_Click;
            button13.Click += PixelButton_Click;
            button14.Click += PixelButton_Click;
            button15.Click += PixelButton_Click;

            // Обработчик кнопки тестового примера
            buttonTest.Click += buttonTest_Click;
        }


        // Обработка нажатия на пиксель
        private void PixelButton_Click(object sender, EventArgs e)
        {
            Button button = (Button)sender;

            // TabIndex используется как индекс пикселя
            int index = button.TabIndex;

            // Переключаем состояние пикселя
            if (button.BackColor == Color.Black)
            {
                button.BackColor = Color.White;
                inputPixels[index] = 0;
            }
            else
            {
                button.BackColor = Color.Black;
                inputPixels[index] = 1;
            }

            // Показываем текущие значения всех 15 пикселей
            labelPixels.Text = string.Join(" ", inputPixels);
        }


        // Сохранение обучающего примера
        private void button_SaveTrainSample_Click(object sender, EventArgs e)
        {
            // Создаём папку Data
            string dataFolder = Path.Combine(
                AppDomain.CurrentDomain.BaseDirectory,
                "Data"
            );

            Directory.CreateDirectory(dataFolder);

            // Путь к файлу обучающих примеров
            string path = Path.Combine(
                dataFolder,
                "train.txt"
            );

            // Первым записываем правильный ответ
            string tmpStr =
                numericUpDown_NecessaryOutput.Value.ToString();

            // Затем записываем 15 пикселей
            for (int i = 0; i < inputPixels.Length; i++)
            {
                tmpStr += " " + inputPixels[i].ToString();
            }

            // Переходим на новую строку
            tmpStr += "\n";

            // Добавляем пример в файл
            File.AppendAllText(path, tmpStr);

            MessageBox.Show(
                "Обучающий пример добавлен!"
            );
        }


        // Сохранение тестового примера
        private void buttonTest_Click(object sender, EventArgs e)
        {
            // Создаём папку Data
            string dataFolder = Path.Combine(
                AppDomain.CurrentDomain.BaseDirectory,
                "Data"
            );

            Directory.CreateDirectory(dataFolder);

            // Путь к файлу тестовых примеров
            string path = Path.Combine(
                dataFolder,
                "test.txt"
            );

            // Создаём строку из 15 пикселей
            string tmpStr = "";

            for (int i = 0; i < inputPixels.Length; i++)
            {
                tmpStr += inputPixels[i].ToString();

                if (i < inputPixels.Length - 1)
                {
                    tmpStr += " ";
                }
            }

            // Переход на новую строку
            tmpStr += "\n";

            // Добавляем тестовый пример в файл
            File.AppendAllText(path, tmpStr);

            MessageBox.Show(
                "Тестовый пример добавлен!"
            );
        }
    }
}