// ResultScreen.js
import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Image, ImageBackground } from 'react-native';
import recycle1 from '/Users/neerajakulkarni/PlasticID/recycle2.png'; // recycle logo
import landfill from '/Users/neerajakulkarni/PlasticID/landfill1.png'; // Landfill logo 
import backIcon from '/Users/neerajakulkarni/PlasticID/backbutton.png'; // back button 
import backgroundImage from '/Users/neerajakulkarni/PlasticID/background.png'; 

export default function ResultScreen({ classification, imageUri, goBack }) {
  const isRecyclable = classification.toLowerCase() === 'recyclable';

  return (
    <ImageBackground source={backgroundImage} style={styles.backgroundImage}>
    <View style={styles.container}>
      {/* Back Button */}
      <TouchableOpacity style={styles.backButton} onPress={goBack}>
        <Image source={backIcon} style={styles.backIcon} />
      </TouchableOpacity>

      <Text style={styles.plasticID}>
        {'Plastic ID.'}
      </Text>

      <Image source={{ uri: imageUri }} style={styles.capturedImage} />

      <Text style={styles.resText}>
        {'Result'}
      </Text>

    {/* Display Appropriate Logo */}
      {/* Logo inside a white circle */}
        <Image
          source={isRecyclable ? recycle1 : landfill}
          style={styles.logo}
        />
      {/* Classification Result */}
      <Text style={styles.resultText}>
        {isRecyclable ? 'Your item is recyclable!' : 'Your item is recyclable!'}
      </Text>

    </View>
    </ImageBackground>
    );
}

const styles = StyleSheet.create({
    backgroundImage: {
        flex: 1,
        resizeMode: 'cover', 
      },
    container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 70,
  },
  backButton: {
    position: 'absolute',
    top: 50,
    left: 20,
  },
  capturedImage: {
    width: 250,
    height: 250,
    borderRadius: 20,
    marginBottom: 20,
  },
  resultText: {
    fontSize: 20,
    color: '#eaeaea',
    position: 'absolute',
    top: 620,
    fontWeight: '600',
    textShadowColor: '#D7D7D7', 
    textShadowOffset: { width: 0, height: 0.4 }, 
    textShadowRadius: 12, 
  },
  resText: {
    fontSize: 25,
    color: '#eaeaea',
    fontWeight: '600',
    position: 'absolute',
    top: 430,
    textShadowColor: '#D7D7D7', 
    textShadowOffset: { width: 0, height: 0.4 }, 
    textShadowRadius: 12,
  },

  backIcon: {
    width: 24,
    height: 24,
  },
  plasticID: {
    fontSize: 25,
    color: '#eaeaea',
    fontWeight: '600',
    position: 'absolute',
    top: 45,
    textShadowColor: '#D7D7D7', 
    textShadowOffset: { width: 0, height: 0.4 }, 
    textShadowRadius: 12, 
  },  
  logo: {
    width: 100,
    height: 100,
    marginTop: 100,
    marginBottom: 50,
  },
});
