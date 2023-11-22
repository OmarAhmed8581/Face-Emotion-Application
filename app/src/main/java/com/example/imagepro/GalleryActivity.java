package com.example.imagepro;

import androidx.appcompat.app.AppCompatActivity;
import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import android.content.Context;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.net.wifi.WifiManager;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import org.opencv.android.Utils;
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

public class GalleryActivity extends AppCompatActivity {
    private Interpreter interpreter;

    private int inputImageSize = 48;
    private int numClasses = 7; // Number of emotion classes
    int YOUR_MODEL_INPUT_HEIGHT=0;
    int YOUR_MODEL_INPUT_WIDTH = 0;

    private int Input_size;
    private int height =0;
    private int width =0;
    private GpuDelegate gpuDelegate = null;
    Button gallery;
    ImageView imageView;
    CascadeClassifier faceCascade;
    private FaceExpressionRecognition faceExpressionRecognition;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_gallery);
        gallery = findViewById(R.id.choosegallery);
        imageView = findViewById(R.id.images);
        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                opengallery();
            }
        });


        Input_size = 48;
        Interpreter.Options options = new Interpreter.Options();
        gpuDelegate = new GpuDelegate();
        options.addDelegate(gpuDelegate);
        options.setNumThreads(4);
        try {
            interpreter = new Interpreter(loadModelFile(getAssets(),"model_optimized.tflite"),options);
            Tensor inputTensor = interpreter.getInputTensor(0);

            // Get the shape of the input tensor
            int[] inputShape = inputTensor.shape();

            // The inputShape will contain the dimensions of the input tensor, e.g., [batchSize, height, width, channels]
            // 'height' is the dimension you're looking for as YOUR_MODEL_INPUT_HEIGHT
            YOUR_MODEL_INPUT_HEIGHT = inputShape[1];
            YOUR_MODEL_INPUT_WIDTH = inputShape[2];
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        try {


            InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalcatface); // Use your own model file
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File cascadeFile = new File(cascadeDir, "haarcascade_frontalcatface_alt");
            FileOutputStream os = new FileOutputStream(cascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;

            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();
            faceCascade = new CascadeClassifier(cascadeFile.getAbsolutePath());
            cascadeDir.delete();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public  void opengallery(){
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        galleryactivity.launch(intent);
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


    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelpath) throws IOException{
        AssetFileDescriptor assetFileDescriptor = assetManager.openFd(modelpath);
        FileInputStream inputStream = new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();

        long startoffset = assetFileDescriptor.getStartOffset();
        long declaredLength = assetFileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startoffset,declaredLength);
    }

    private Bitmap cropCenter(Bitmap bitmap, int x, int y, int w, int h) {
        return Bitmap.createBitmap(bitmap, x, y, w, h);
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

    ActivityResultLauncher<Intent> galleryactivity = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(),
            new ActivityResultCallback<ActivityResult>() {
                @Override
                public void onActivityResult(ActivityResult result) {
                    Uri uri_image = result.getData().getData();
//

                    Bitmap bitmapImage = null;
                    try {
                        bitmapImage = BitmapFactory.decodeStream(getContentResolver().openInputStream(uri_image));
                    } catch (Exception e) {
                        e.printStackTrace();
                    }

// Check if the Bitmap is not null
                    if (bitmapImage != null) {
                        // Convert the Bitmap to a grayscale Mat
                        Mat matImage = new Mat();
                        Utils.bitmapToMat(bitmapImage, matImage);
                        Mat grayMat = new Mat();
                        Imgproc.cvtColor(matImage, grayMat, Imgproc.COLOR_BGR2GRAY);

                        // Perform face detection
                        MatOfRect faces = new MatOfRect();
                        faceCascade.detectMultiScale(grayMat, faces, 1.1, 2, 2, new Size(30, 30), new Size());

                        // Draw rectangles around the detected faces
                        Rect[] facesArray = faces.toArray();
                        for (Rect face : facesArray) {
                            Imgproc.rectangle(matImage, face.tl(), face.br(), new Scalar(255, 0, 0), 4);
                            Rect rect = new Rect((int)face.tl().x,(int)face.tl().y,((int)face.br().x)-(int)(face.tl().x),
                                    ((int)face.br().y)-(int)(face.tl().y));
                            Mat croppedimages = new Mat(matImage,rect);
                            Bitmap bitmap = null;
                            bitmap = Bitmap.createBitmap(croppedimages.cols(),croppedimages.rows(),Bitmap.Config.ARGB_8888);
                            Utils.matToBitmap(croppedimages,bitmap);
                            Bitmap scale = Bitmap.createScaledBitmap(bitmap,48,48,false);
                            String emotiontext = detectEmotion(bitmap,(int)face.tl().x,(int)face.tl().y,((int)face.br().x)-(int)(face.tl().x),((int)face.br().y)-(int)(face.tl().y));
                            Imgproc.putText(matImage,emotiontext,new Point((int)face.tl().x+10,(int)face.tl().y+20),
                                    1,4.5,new Scalar(0,0,255,150),5);
                        }



                        // Convert the result Mat back to a Bitmap
                        Bitmap resultBitmap = Bitmap.createBitmap(matImage.cols(), matImage.rows(), Bitmap.Config.ARGB_8888);
                        Utils.matToBitmap(matImage, resultBitmap);



                        // Display the result Bitmap in the ImageView
                        imageView.setImageBitmap(resultBitmap);
                    }



                }
            });
}