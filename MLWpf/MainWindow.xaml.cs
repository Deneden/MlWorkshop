using Microsoft.Win32;
using System;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Media.Imaging;

namespace MLWpf
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void OpenImageButton(object sender, RoutedEventArgs e)
        {
            var openFile = new OpenFileDialog
            {
                Title = "Выберите изображение",
                Filter = "Изображения|*.png;*.jpg|Все файлы|*.*",
                CheckFileExists= true
            };

            if (openFile.ShowDialog() != true) return;

            var fileName = openFile.FileName;

            ImgViewer.Source = new BitmapImage(new Uri(fileName));

            var file = File.ReadAllBytes(fileName);
            
            var sampleData = new RickAndMortyTrain.ConsoleApp.RickAndMortyTrain.ModelInput
            {
                ImageSource = file
            };
            
            var predictionResult = RickAndMortyTrain.ConsoleApp.RickAndMortyTrain.Predict(sampleData);

            PredictionText.Text = $"{predictionResult.PredictedLabel} - {predictionResult.Score.Max():p0}";
        }
    }
}
