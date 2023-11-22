package com.example.imagepro;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;
import android.widget.Toast;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class FaceExpressionRecognition {



    private Interpreter interpreter;
    private int Input_size;
    private int height =0;
    private int width =0;
    private GpuDelegate gpuDelegate = null;
    private CascadeClassifier cascadeClassifier;
    private Context context1;

    private int numClasses = 7; // Number of emotion classes
    int YOUR_MODEL_INPUT_HEIGHT=0;
    int YOUR_MODEL_INPUT_WIDTH = 0;
    private List emotion_list =new ArrayList();


    FaceExpressionRecognition(AssetManager assetManager , Context context, String modelpath , int inputsize) throws IOException {
        context1 = context;
        Input_size = inputsize;
        Interpreter.Options options = new Interpreter.Options();
        gpuDelegate = new GpuDelegate();
        options.addDelegate(gpuDelegate);
        options.setNumThreads(4);
        interpreter = new Interpreter(loadModelFile(assetManager,modelpath),options);
        Tensor inputTensor = interpreter.getInputTensor(0);

        // Get the shape of the input tensor
        int[] inputShape = inputTensor.shape();

        // The inputShape will contain the dimensions of the input tensor, e.g., [batchSize, height, width, channels]
        // 'height' is the dimension you're looking for as YOUR_MODEL_INPUT_HEIGHT
        YOUR_MODEL_INPUT_HEIGHT = inputShape[1];
        YOUR_MODEL_INPUT_WIDTH = inputShape[2];


        try{
            InputStream inputStream = context.getResources().openRawResource(R.raw.haarcascade_frontalcatface);
            File cascade = context.getDir("cascade",Context.MODE_PRIVATE);
            File mCascadefile = new File(cascade , "haarcascade_frontalcatface_alt");
            FileOutputStream fileOutputStream = new FileOutputStream(mCascadefile);
            byte[] buffer= new byte[4096];
            int byteRead;
            while ((byteRead = inputStream.read(buffer))!=-1){
                fileOutputStream.write(buffer,0,byteRead);

            }
            inputStream.close();
            fileOutputStream.close();
            cascadeClassifier = new CascadeClassifier(mCascadefile.getAbsolutePath());
        }
        catch (IOException e){
            e.printStackTrace();
        }
    }


    private Bitmap preprocessImage(Bitmap bitmap) {
        return Bitmap.createScaledBitmap(bitmap, YOUR_MODEL_INPUT_WIDTH, YOUR_MODEL_INPUT_HEIGHT, true);
    }

    private String getEmotionLabel(int predictedClass) {
        String[] emotionLabels = {"anger", "disgust", "fear", "happy", "neutral", "sadness", "surprised"};

        if (predictedClass >= 0 && predictedClass < emotionLabels.length) {
            return emotionLabels[predictedClass];
        } else {
            return "unknown";
        }
    }

    public String detectEmotion(Bitmap raw, int x, int y, int w, int h) {
//        Bitmap croppedImage = cropCenter(raw, x, y, w, h);
        Bitmap preprocessedImage = preprocessImage(raw);
        ByteBuffer byteBuffer = convertBitmaptoBytes(preprocessedImage);

        float[][] result = new float[1][numClasses];
        interpreter.run(byteBuffer, result);

        int predictedClass = argmax(result[0]);
        return getEmotionLabel(predictedClass)+"( "+String.valueOf(predictedClass)+")";
    }

    private int argmax(float[] array) {
        int maxIdx = 0;
        float maxVal = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxVal) {
                maxVal = array[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }



    public Mat recoginzeImage(Mat mat_images) {
        Core.flip(mat_images.t(), mat_images, 1);
        Mat grayscale = new Mat();
        Imgproc.cvtColor(mat_images, grayscale, Imgproc.COLOR_RGB2GRAY);
        height = grayscale.height();
        width = grayscale.width();
        int absolutionfacesize = (int) (height * 0.1);
        MatOfRect faces = new MatOfRect();
        if (cascadeClassifier != null) {
            cascadeClassifier.detectMultiScale(grayscale, faces, 1.1, 2, 2,
                    new Size(absolutionfacesize, absolutionfacesize), new Size());
        }
        Rect[] facearray = faces.toArray();
        int fontFace = Core.FONT_HERSHEY_SIMPLEX;
        double fontScale = 1.0;
        int thickness = 2;
        Scalar textColor = new Scalar(0, 0, 255, 150); // Red text with transparency
        Scalar bgColor = new Scalar(255, 255, 255, 150);

        if(facearray.length==0){
            if(emotion_list.size()>0){
                emotion_list.clear();
            }
        }


        for (int i = 0; i < facearray.length; i++) {
            Imgproc.rectangle(mat_images, facearray[i].tl(), facearray[i].br(), new Scalar(0, 255, 0, 144), 2);
            Rect rect = new Rect((int) facearray[i].tl().x, (int) facearray[i].tl().y, ((int) facearray[i].br().x) - (int) (facearray[i].tl().x), ((int) facearray[i].br().y) - (int) (facearray[i].tl().y));
            Mat croppedimages = new Mat(mat_images, rect);
            Bitmap bitmap = null;
            bitmap = Bitmap.createBitmap(croppedimages.cols(), croppedimages.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(croppedimages, bitmap);
            Bitmap scale = Bitmap.createScaledBitmap(bitmap, 48, 48, false);
            ByteBuffer byteBuffer = convertBitmaptoBytes(scale);
            String emotiontext = detectEmotion(bitmap,(int)facearray[i].tl().x,(int)facearray[i].tl().y,((int)facearray[i].br().x)-(int)(facearray[i].tl().x),((int)facearray[i].br().y)-(int)(facearray[i].tl().y));
//            if (emotion_list.size() == 0) {
//                Imgproc.putText(mat_images, emotiontext, new Point((int) facearray[i].tl().x + 10, (int) facearray[i].tl().y + 20), 1, 1, new Scalar(0, 0, 255, 150), 2);
//                List temp = new ArrayList();
//                temp.add(emotiontext);
//                temp.add((int) facearray[i].tl().x + 10);
//                temp.add((int) facearray[i].tl().y + 20);
//                emotion_list.add(temp);
//            } else {
//                if (facearray.length == emotion_list.size()) {
//
//                    List a = (List) emotion_list.get(i);
//                    Imgproc.putText(mat_images, String.valueOf(a.get(0)), new Point((int) facearray[i].tl().x + 10, (int) facearray[i].tl().y + 20), 1, 1, new Scalar(0, 0, 255, 150), 2);
//                } else {
//                    Imgproc.putText(mat_images, emotiontext, new Point((int) facearray[i].tl().x + 10, (int) facearray[i].tl().y + 20), 1, 1, new Scalar(0, 0, 255, 150), 2);
//                    emotion_list.clear();
//                    List temp = new ArrayList();
//                    temp.add(emotiontext);
//                    temp.add((int) facearray[i].tl().x + 10);
//                    temp.add((int) facearray[i].tl().y + 20);
//                    emotion_list.add(temp);
//                }
//            }
            Imgproc.putText(mat_images, emotiontext, new Point((int) facearray[i].tl().x + 10, (int) facearray[i].tl().y + 20), 1, 1, new Scalar(0, 0, 255, 150), 2);
        }
        Core.flip(mat_images.t(), mat_images, 0);
        return mat_images;
    }
    private  String get_emotion_text(float emotion_v){
        String val="";
        if(emotion_v>0 & emotion_v<0.5){
            val="Surprise";
        }
        else if(emotion_v>=0.5 & emotion_v<1.5){
            val="Fear";
        }

        else if(emotion_v>=1.5 & emotion_v<2.5){
            val="Angry";
        }
        else if(emotion_v>=2.5 & emotion_v<3.5){
            val="Neutral";
        }
        else if(emotion_v>=3.5 & emotion_v<4.5){
            val="Sad";
        }
        else if(emotion_v>=4.5 & emotion_v<5.5){
            val="Disgust";
        }
        else{
            val="Happy";
        }
        return  val;
    }

    private ByteBuffer convertBitmaptoBytes(Bitmap scalebitmap){
        ByteBuffer byteBuffer;
        int size_images = Input_size;
        byteBuffer = ByteBuffer.allocateDirect(4*1*YOUR_MODEL_INPUT_HEIGHT*YOUR_MODEL_INPUT_WIDTH*3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intvalue = new int[YOUR_MODEL_INPUT_HEIGHT*YOUR_MODEL_INPUT_WIDTH];
        scalebitmap.getPixels(intvalue,0,scalebitmap.getWidth(),0,0,scalebitmap.getWidth(),scalebitmap.getHeight());
        int pixel=0;
        for(int i=0;i<YOUR_MODEL_INPUT_HEIGHT;++i){
            for(int j=0;j<YOUR_MODEL_INPUT_WIDTH;++j){
                final int val = intvalue[pixel++];
                byteBuffer.putFloat((((val>>16)&0xFF))/255.0f);
                byteBuffer.putFloat((((val>>8)&0xFF))/255.0f);
                byteBuffer.putFloat(((val & 0xFF))/255.0f);
            }
        }

        return byteBuffer;
    }



    private MappedByteBuffer loadModelFile(AssetManager assetManager,String modelpath) throws IOException{
        AssetFileDescriptor assetFileDescriptor = assetManager.openFd(modelpath);
        FileInputStream inputStream = new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();

        long startoffset = assetFileDescriptor.getStartOffset();
        long declaredLength = assetFileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startoffset,declaredLength);
    }



}
