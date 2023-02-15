using Microsoft.ML;
using Microsoft.ML.Data;
using MLConsole;

var projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
var assetsPath = Path.Combine(projectDirectory, "assets");
var imagesFolder = Path.Combine(assetsPath, "images");
var trainTagsTsv = Path.Combine(imagesFolder, "tags.tsv");
var testTagsTsv = Path.Combine(imagesFolder, "test-tags.tsv");
var predictSingleImage = Path.Combine(imagesFolder, "toaster3.jpg");
var inceptionTensorFlowModel = Path.Combine(assetsPath, "inception", "tensorflow_inception_graph.pb");

//Входная точка для всех ML операций, аналог DbContext
MLContext mlContext = new MLContext();

//Интерфейс набора трансформаций применяемый к модели. Работает лениво (аналог IQueryable)
ITransformer model = GenerateModel(mlContext);

ClassifySingleImage(mlContext, model);

void ClassifySingleImage(MLContext context, ITransformer transformer)
{
    var imageData = new ImageData()
    {
        ImagePath = predictSingleImage
    };

    var predictor = context.Model.CreatePredictionEngine<ImageData, ImagePrediction>(transformer);
    var prediction = predictor.Predict(imageData);

    Console.WriteLine(
        $"Image: {Path.GetFileName(imageData.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");
}

ITransformer GenerateModel(MLContext context)
{
    IEstimator<ITransformer> pipeline = context.Transforms
            //загрузить изображения
        .LoadImages(outputColumnName: "input", imageFolder: imagesFolder, inputColumnName: nameof(ImageData.ImagePath))
            //изменить размер изображения под стандарт для TensorFlow Inception
        .Append(context.Transforms.ResizeImages(outputColumnName: "input", imageWidth: InceptionSettings.ImageWidth,
            imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
            //извлечь значения пикселей из данных и отформатировать их (чередование, смещение)
        .Append(context.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsLast,
            offsetImage: InceptionSettings.Mean))
            //загрузка и оценка набора данных моделью TensorFlow
            //softmax_2_preactivation - данные предпоследнего слоя являющегося вектором признаков
            //https://microscope.openai.com/models/inceptionv1?models.technique=deep_dream
        .Append(context.Model.LoadTensorFlowModel(inceptionTensorFlowModel).ScoreTensorFlowModel(
            outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
            //преобразует строковые метки в числовые ключи т.к. модели работают с числовыми форматами
        .Append(context.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label"))
            //трейнер для классификации работающий по алгоритму L-BFGS https://habr.com/ru/post/333356/
            .Append(context.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "LabelKey",
            featureColumnName: "softmax2_pre_activation"))
            //преобразует числовые ключи в текстовые метки
        .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
            //добавление кэширования данных, полезно при нескольких проходах обучения
        .AppendCacheCheckpoint(context);

    //Интерфейс для предоставления конвеера данных в ML операциях, аналог IEnumerable. 
    IDataView trainingData = context.Data.LoadFromTextFile<ImageData>(path: trainTagsTsv, hasHeader: false);
    
    ITransformer model = pipeline.Fit(trainingData);
    
    //сохраняем модель для дальнейшего переиспользования
    mlContext.Model.Save(model, trainingData.Schema, "model.zip");
    
    //Загружаем тестовые файлы по файлу тегов
    IDataView testData = mlContext.Data.LoadFromTextFile<ImageData>(path: testTagsTsv, hasHeader: false);

    IDataView predictions = model.Transform(testData);

    IEnumerable<ImagePrediction> imagePredictionData = context.Data.CreateEnumerable<ImagePrediction>(predictions, true);
    
    DisplayResults(imagePredictionData);

    MulticlassClassificationMetrics metrics =
        context.MulticlassClassification.Evaluate(predictions,
            labelColumnName: "LabelKey",
            predictedLabelColumnName: "PredictedLabel");

    Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
    Console.WriteLine($"PerClassLogLoss is: {String.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");

    var trainingDebug = model.Preview(trainingData);
    var testDebug = model.Preview(testData);
    
    return model;
}

void DisplayResults(IEnumerable<ImagePrediction> imagePredictionData)
{
    foreach (ImagePrediction prediction in imagePredictionData)
    {
        Console.WriteLine(
            $"Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");
    }
}

struct InceptionSettings //Параметры для TensorFlow модели Inception
{
    public const int ImageHeight = 224;
    public const int ImageWidth = 224;
    public const float Mean = 117;
    public const bool ChannelsLast = true;
}