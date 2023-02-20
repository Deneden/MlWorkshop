using GPTConsole;
using Microsoft.ML;
using Microsoft.ML.Transforms.Image;
using Microsoft.ML.Vision;

//GPT Comment:
//Load the image data from the folders using the LoadFromEnumerable method of the MLContext class.
var mlContext = new MLContext();

var images = mlContext.Data.LoadFromEnumerable<ImageData>(
    Directory.GetFiles("cats").Select(file => new ImageData { ImagePath = file, Label = "cat" })
        .Concat(Directory.GetFiles("dogs").Select(file => new ImageData { ImagePath = file, Label = "dog" })));

//GPT Comment:
//Split the data into training and testing datasets using the TrainTestSplit method.
var dataSplit = mlContext.Data.TrainTestSplit(images, testFraction: 0.2);

//GPT Comment:
//Define a pipeline to preprocess the image data and train the model.
var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
    .Append(mlContext.Transforms.LoadRawImageBytes("Image", null, "ImagePath"))
    //replace to:
    //.Append(mlContext.Transforms.ResizeImages("Image", ImageResizingEstimator.Defaults.ImageResizingEstimatorSettings("Image", 224, 224,"Image"))
    .Append(mlContext.Transforms.ResizeImages("Image", ImageResizingEstimator.Defaults.ImageResizingEstimatorSettings("Image", 224, 224)))
    .Append(mlContext.Transforms.ExtractPixels("Image", interleavePixelColors: true))
    //replace to:
    //.Append(mlContext.MulticlassClassification 
    .Append(mlContext.Model 
        .Trainers
        .ImageClassification(new ImageClassificationTrainer.Options { Epoch = 50, Arch = ImageClassificationTrainer.Architecture.ResnetV2101 })
        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

var model = pipeline.Fit(dataSplit.TrainSet);

//GPT Comment:
//Use the trained model to make predictions on the testing dataset using the Transform method.
var predictions = model.Transform(dataSplit.TestSet);

var metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "Label", predictedLabelColumnName: "PredictedLabel");

Console.WriteLine($"Accuracy: {metrics.MacroAccuracy}");
