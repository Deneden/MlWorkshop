using System.Numerics;
using Microsoft.ML.Data;
using Tensorflow;

namespace MLConsole;

public class ImagePrediction : ImageData
{
    public float[] Score;

    public string PredictedLabelValue;
}