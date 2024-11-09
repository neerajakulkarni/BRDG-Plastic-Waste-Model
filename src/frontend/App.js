import React, {useState} from 'react';
import {SafeAreaView, View, Text, Button, Image, StyleSheet, Alert} from 'react-native';
import {RNCamera} from 'react-native-camera';
import axios from 'axios';
import { TouchableOpacity } from 'react-native';
import ResultScreen from '/Users/neerajakulkarni/PlasticID/ResultScreen.js'; // Ensure the path is correct


export default function App() {
  const [imageUri, setImageUri] = useState(null);  // Store the image path
  const [classification, setClassification] = useState('');  // Store the classification result

  // Function to take a picture
  const takePicture = async (camera) => {
    try {
        const options = { quality: 0.5, base64: true };
        const data = await camera.takePictureAsync(options);
        setImageUri(data.uri);
        
        // Send the image URI to the model
        sendImageToModel(data.uri); // Pass the URI here
    } catch (error) {
        Alert.alert('Error', 'Failed to take picture');
    }
};

  // Function to send the image to the Python backend
  const sendImageToModel = async (uri) => {
    try {
        // Create a FormData object to hold the image
        const formData = new FormData();
        formData.append('file', {
            uri: uri,
            type: 'image/jpeg', // Adjust based on your image format
            name: 'photo.jpg',   // You can also use data.filename if needed
        });

        const response = await axios.post('http://192.168.0.202:5000/predict', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });

        // Process the response to display classification result
        setClassification(response.data.predictions); // Get the top prediction
    } catch (error) {
        console.error('Error uploading image: ', error);
        Alert.alert('Error', 'Failed to classify image');
    }
};

  // Function to go back to the camera
  const goBack = () => {
    setImageUri(null);
    setClassification('');
  };

  return (
    <SafeAreaView style={styles.container}>
      {imageUri ? (
        <ResultScreen classification={classification} imageUri={imageUri} goBack={goBack} />
        ) : (
        <RNCamera
          style={styles.camera}
          type={RNCamera.Constants.Type.back}
          captureAudio={false}
          androidCameraPermissionOptions={{
            title: 'Permission to use camera',
            message: 'We need your permission to use your camera',
            buttonPositive: 'Ok',
            buttonNegative: 'Cancel',
          }}
        >
          {({ camera, status }) => {
            if (status !== 'READY') return <Text>Loading...</Text>;
            return (
                <TouchableOpacity 
                style={styles.shutterButton} 
                onPress={() => takePicture(camera)}
                >
                <View style={styles.innerShutter} />
                </TouchableOpacity>
            );
          }}
        </RNCamera>
      )}
    </SafeAreaView>
  );
}


const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#fff',
  },
  camera: {
    flex: 1,
    width: '100%',
  },
  captureButton: {
    flex: 0,
    position: 'absolute',
    bottom: 20,
    alignSelf: 'center',
  },
  previewContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  previewImage: {
    width: 300,
    height: 300,
  },
  resultText: {
    fontSize: 20,
    marginTop: 20,
  },
  shutterButton: {
    position: 'absolute',
    bottom: 30,
    alignSelf: 'center', 
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#000',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 30,
    borderWidth: 5,
    borderColor: '#fff',
  },
  innerShutter: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#fff',
  },
  
});

