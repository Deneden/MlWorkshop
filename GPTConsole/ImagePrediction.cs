using Microsoft.ML.Data;

namespace GPTConsole;

//Define a class to represent the image data.
//This class should have properties for the image path and the label (cat or dog).

public class ImagePrediction
{
    [ColumnName("PredictedLabel")]
    public string PredictedLabel;
}