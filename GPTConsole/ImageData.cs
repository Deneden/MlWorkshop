using Microsoft.ML.Data;

namespace GPTConsole;

//Define a class to represent the predicted output.
//This class should have a property for the predicted label.

public class ImageData
{
    [LoadColumn(0)]
    public string ImagePath;

    [LoadColumn(1)]
    public string Label;
}