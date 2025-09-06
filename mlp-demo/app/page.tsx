'use client'

import React, { useState, useRef, useCallback } from 'react';
import DrawingPad, { PixelGrid, PixelUpdate } from "@/components/DrawingPad"
import Button from '@/components/Button';

// Type definitions for the API
interface PredictRequest {
  array: PixelGrid;
}

interface PredictResponse {
  status: string;
  input_array: number[][];
  flattened_length: number;
  probabilities: number[];
  predicted_class: number;
  confidence: number;
}

interface ErrorResponse {
  error: string;
  message: string;
  example?: { array: number[][] };
  shape?: number[];
}


export default function Home() {
  const [pixels, setPixels] = useState<PixelGrid>(() => 
    Array(28).fill(null).map(() => Array(28).fill(0))
  );

  const [tempClass, setTempClass] = useState<number>(0);
  const [tempConfidence, setTempConfidence] = useState<number>(1);
  const [brushSize, setBrushSize] = useState<number>(4);
  const [currentPred, setPred] = useState<PredictResponse | null>(null);
  

  const handlePixelUpdate = useCallback((updates: PixelUpdate[]) => {
    setPixels(prevPixels => {
      const newPixels: PixelGrid = prevPixels.map(row => [...row]);
      updates.forEach(({ row, col, value }) => {
        newPixels[row][col] = value;
      });
      return newPixels;
    });
  }, []);

  async function predictArray(array2d: PixelGrid): Promise<PredictResponse> {
    try {
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          array: array2d
        } as PredictRequest)
      });

      const data = await response.json();

      if (!response.ok) {
        const errorData = data as ErrorResponse;
        throw new Error(`API Error ${response.status}: ${errorData.message}`);
      }

      return data as PredictResponse;
    } catch (error) {
      if (error instanceof Error) {
        throw new Error(`Failed to make prediction: ${error.message}`);
      }
      throw new Error('Failed to make prediction: Unknown error');
    }
  }

  const handlePrediction = async () => {
    const newPred = await predictArray(pixels)
    if (currentPred !== null) {
      setTempClass(currentPred.predicted_class)
      setTempClass(currentPred.confidence)
    }

    setPred(newPred)
    
  }
 

  const clearPixels = (): void => {

    if (currentPred !== null) {
      setTempClass(currentPred.predicted_class)
      setTempConfidence(currentPred.confidence)
    }

    setPixels(Array(28).fill(null).map(() => Array(28).fill(0)))
    
    setTimeout(() => {
      setPred(null)
    }, 300) // Wait for fade out animation
  }

  return (
  
    <div className="min-h-screen flex flex-col items-center mt-50 p-4 gap-2">
      <div className='text-8xl font-bold text-amber-50 text-center mb-20'>
        <h1>
           MNIST Digit Predictor
        </h1>
      </div>

      <div className="rounded-md p-4 bg-stone-800  aspect-square w-96">
        <DrawingPad 
                pixels={pixels}
                brushSize={brushSize}
                onPixelUpdate={handlePixelUpdate}
              />
      </div>
     

      <div className = "flex flex-row w-80 justify-between p-4">
        
        <Button
          color="#2563eb"
          children="Clear"
          onClick={clearPixels}
        />
        <Button
          color="#52c465"
          children="Predict"
          onClick={handlePrediction}
        />
      </div>

       {/* Animated div that fades in/out with upward movement */}
        <div 
          className={`
            transition-all duration-500 ease-in-out
            ${currentPred !== null
              ? 'opacity-100 transform translate-y-0' 
              : 'opacity-0 transform translate-y-4 pointer-events-none'
            }
          `}
        >
          <div className="bg-stone-900 p-4 rounded-lg w-50 aspect-square shadow-lg border border-stone-900 flex flex-col justify-between">
            <h1 className="text-emerald-300 text-9xl font-extrabold">
              {currentPred?.predicted_class ?? tempClass}
            </h1>
            <p className='text-gray-300 font-medium font font-mono'>
              {currentPred !== null ? (currentPred.confidence * 100).toFixed(2) : (tempConfidence * 100).toFixed(2)}% confidence
            </p>
          </div>
        </div>
    </div>
  );
}
