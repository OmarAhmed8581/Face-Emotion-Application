package com.example.imagepro;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraInfo;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.Preview;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Camera;
import android.media.FaceDetector;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;

import com.google.common.util.concurrent.ListenableFuture;
//import com.google.android.gms.vision.CameraSource;
//import com.google.android.gms.vision.Detector;
//import com.google.android.gms.vision.face.Face;
//import com.google.android.gms.vision.face.FaceDetector;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.concurrent.ExecutionException;

public class CameraActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2{
    private static final String TAG="MainActivity";



    private Mat mRgba;
    private Mat mGray;
    private CameraBridgeViewBase mOpenCvCameraView;
    private FaceExpressionRecognition faceExpressionRecognition;
    private BaseLoaderCallback mLoaderCallback =new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status){
                case LoaderCallbackInterface
                        .SUCCESS:{
                    Log.i(TAG,"OpenCv Is loaded");
                    mOpenCvCameraView.enableView();
                }
                default:
                {
                    super.onManagerConnected(status);

                }
                break;
            }
        }
    };


    private JavaCameraView javaCameraView;
    private int currentCameraId = CameraBridgeViewBase.CAMERA_ID_BACK;
    private ImageView switchcamera;

    // ... Your existing code ...

    public void onSwitchCameraButtonClick() {
        // Switch between front and back cameras
        currentCameraId = (currentCameraId == CameraBridgeViewBase.CAMERA_ID_BACK) ?
                CameraBridgeViewBase.CAMERA_ID_FRONT : CameraBridgeViewBase.CAMERA_ID_BACK;
        mOpenCvCameraView.disableView();
        mOpenCvCameraView.setCameraIndex(currentCameraId);
        mOpenCvCameraView.enableView();
    }


    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;


    public CameraActivity(){
        Log.i(TAG,"Instantiated new "+this.getClass());
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        int MY_PERMISSIONS_REQUEST_CAMERA=0;
        // if camera permission is not given it will ask for it on device
        if (ContextCompat.checkSelfPermission(CameraActivity.this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(CameraActivity.this, new String[] {Manifest.permission.CAMERA}, MY_PERMISSIONS_REQUEST_CAMERA);
        }

        setContentView(R.layout.activity_camera);

        mOpenCvCameraView=(CameraBridgeViewBase) findViewById(R.id.frame_Surface);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        cameraProviderFuture = ProcessCameraProvider.getInstance(this);

//        FaceDetector detector = new FaceDetector.Builder(getApplicationContext())
//                .setTrackingEnabled(false)
//                .setClassificationType(FaceDetector.ALL_CLASSIFICATIONS)
//                .build();
//        if (!detector.isOperational()) {
//            // Handle error if the detector is not operational
//            return;
//        }

        switchcamera  =findViewById(R.id.switchCameraButton);
        switchcamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                onSwitchCameraButtonClick();
            }
        });
        try{
            faceExpressionRecognition = new FaceExpressionRecognition(getAssets(),CameraActivity.this,"model_optimized.tflite",48);

        }catch (IOException e){
            e.printStackTrace();
        }


    }


    private int getCameraRotation() {
        // Retrieve the camera orientation here using the CameraX API
        // You need to implement this part based on your CameraX setup
        // For the sake of example, I'm assuming you get the camera orientation from CameraX as rotation degrees.

        int cameraOrientationDegrees = 90; // Replace with the actual camera orientation

        int rotation = 0;
        switch (cameraOrientationDegrees) {
            case 90:
                rotation = Core.ROTATE_90_CLOCKWISE;
                break;
            case 180:
                rotation = Core.ROTATE_180;
                break;
            case 270:
                rotation = Core.ROTATE_90_COUNTERCLOCKWISE;
                break;
        }
        return rotation;
    }


    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()){
            //if load success
            Log.d(TAG,"Opencv initialization is done");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        else{
            //if not loaded
            Log.d(TAG,"Opencv is not loaded. try again");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0,this,mLoaderCallback);
        }
    }
    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView !=null){
            mOpenCvCameraView.disableView();
        }
    }
    public void onDestroy(){
        super.onDestroy();
        if(mOpenCvCameraView !=null){
            mOpenCvCameraView.disableView();
        }

    }

    public void onCameraViewStarted(int width ,int height){
        mRgba=new Mat(height,width, CvType.CV_8UC4);
        mGray =new Mat(height,width,CvType.CV_8UC1);
    }
    public void onCameraViewStopped(){
        mRgba.release();
    }
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame){
        mRgba=inputFrame.rgba();
        mGray=inputFrame.gray();

        if (currentCameraId == CameraBridgeViewBase.CAMERA_ID_FRONT ) {
            // Perform the rotation using Core.rotate() for the front camera
            Mat rotatedFrame = new Mat();
            Core.rotate(mRgba, rotatedFrame, Core.ROTATE_180);
            mRgba = rotatedFrame.clone();
            rotatedFrame.release();
        }
        mRgba = faceExpressionRecognition.recoginzeImage(mRgba);
        return mRgba;

    }

}