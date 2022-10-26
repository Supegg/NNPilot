using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace OpencvPilot
{
    class Program
    {
        static void Main(string[] args)
        {
            const string protoTxt = @"Data\MobileNetSSD_deploy.prototxt";
            const string caffeModel = @"Data\MobileNetSSD_deploy.caffemodel";
            const string synsetWords = @"Data\MobileNetSSD_deploy.class";
            string[] classes = File.ReadAllLines(synsetWords);

            VideoCapture capture = new VideoCapture();
            int blob_height = 300; // 高度缩放到300
            int blob_width = 0;
            double color_scale = 1.0 / 127.5; // 颜色缩放
            Scalar average_color = new Scalar(127.5, 127.5, 127.5); // 均值 mean
            double confidence_threshold = 0.5; // 最小可信阈值

            using var net = CvDnn.ReadNetFromCaffe(protoTxt, caffeModel); // 读取模型
            Console.WriteLine("Layer names: {0}", string.Join(",\t", net.GetLayerNames()));
            Console.WriteLine();

            // Convert Mat to batch of images
            Mat frame = new Mat();
            capture.Open(0, VideoCaptureAPIs.ANY); // 开启摄像头
            while (true)
            {
                frame = capture.RetrieveMat();
                if (frame.Empty())
                {
                    break;
                }

                blob_width = (int)(1.0 * blob_height * frame.Width / frame.Height); // 等比缩放
                using var inputBlob = CvDnn.BlobFromImage(frame, color_scale, new Size(blob_width, blob_height), average_color);
                net.SetInput(inputBlob, "data"); // 检查 input layer，获取参数名
                using var prob = net.Forward(); //net.Forward("detection_out");

                // find the objects
                // 检查 output layer，分析输出数据
                using var probMat = prob.Reshape(1, 100); //Reshape(1, 1);
                float[] predict = new float[7];
                for (int i = 0; i < probMat.Rows; i++)
                {
                    for (int j = 0; j < probMat.Cols; j++)
                    {
                        predict[j] = probMat.At<float>(i, j);
                    }

                    if (predict[1] == 0)
                    {
                        break;
                    }

                    //Console.WriteLine(predict[0]); // always 0
                    //Console.WriteLine(predict[1]); // 目标索引，从1开始
                    //Console.WriteLine(predict[2]); // confidence

                    int x0 = (int)(frame.Width * predict[3]);
                    int y0 = (int)(frame.Height * predict[4]);
                    int x1 = (int)(frame.Width * predict[5]);
                    int y1 = (int)(frame.Height * predict[6]);
                    Cv2.Rectangle(frame, new Point(x0, y0), new Point(x1, y1), new Scalar(0, 255, 0));
                    Cv2.PutText(frame, $"{classes[(int)predict[1] - 1]}, {predict[2]:f4}", new Point(x0, y0),
                        HersheyFonts.HersheySimplex, 1, new Scalar(0, 255, 0));
                    Cv2.ImShow("MobileNetSSD", frame);

                }
                if (Cv2.WaitKey(40) == 27) // Escape
                {
                    Cv2.DestroyAllWindows();
                    break;
                }
            }

            Console.WriteLine("Press any key to exit");
            Console.Read();
        }

        /// <summary>
        /// Find best class for the blob (i. e. class with maximal probability)
        /// 适用于只输出 confidence 的结果
        /// </summary>
        /// <param name="probBlob"></param>
        /// <param name="classId"></param>
        /// <param name="classProb"></param>
        private static void GetMaxClass(Mat probBlob, out int classId, out double classProb)
        {
            // reshape the blob to 1*n matrix
            using var probMat = probBlob.Reshape(1, 1);
            Cv2.MinMaxLoc(probMat, out _, out classProb, out _, out var classNumber);
            classId = classNumber.X;
        }
    }
}
